# PyTorch T5 model
# This code is modified based on modeling_t5.py in Huggingface Transformers.

import copy
import inspect
import logging
import math
import os
import warnings
from collections import UserDict
from typing import Callable, List, Optional, Union, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.nn.utils.rnn import pad_sequence
from torch.utils.checkpoint import checkpoint
from torch_scatter import scatter_mean
from transformers import T5Config
from transformers.file_utils import DUMMY_INPUTS, DUMMY_MASK
from transformers.generation.beam_constraints import (DisjunctiveConstraint, PhrasalConstraint)
from transformers.generation.beam_search import BeamScorer, ConstrainedBeamSearchScorer
from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import (StoppingCriteriaList, validate_stopping_criteria, )
from transformers.generation.utils import (
    GreedySearchEncoderDecoderOutput, GreedySearchDecoderOnlyOutput, SampleEncoderDecoderOutput,
    SampleDecoderOnlyOutput, BeamSearchEncoderDecoderOutput, BeamSearchDecoderOnlyOutput,
    BeamSampleEncoderDecoderOutput, BeamSampleDecoderOnlyOutput, ContrastiveSearchEncoderDecoderOutput,
    ContrastiveSearchDecoderOnlyOutput)
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, BaseModelOutput, Seq2SeqLMOutput
from transformers.modeling_utils import PreTrainedModel, find_pruneable_heads_and_indices, prune_linear_layer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils import is_torch_fx_proxy, add_start_docstrings
from transformers.utils.model_parallel_utils import assert_device_map, get_device_map

from optimal_trans_thu import optimal_transport_dist

GreedySearchOutput = Union[GreedySearchEncoderDecoderOutput, GreedySearchDecoderOnlyOutput]
SampleOutput = Union[SampleEncoderDecoderOutput, SampleDecoderOnlyOutput]
BeamSearchOutput = Union[BeamSearchEncoderDecoderOutput, BeamSearchDecoderOnlyOutput]
BeamSampleOutput = Union[BeamSampleEncoderDecoderOutput, BeamSampleDecoderOnlyOutput]
ContrastiveSearchOutput = Union[ContrastiveSearchEncoderDecoderOutput, ContrastiveSearchDecoderOnlyOutput]
GenerateOutput = Union[GreedySearchOutput, SampleOutput, BeamSearchOutput, BeamSampleOutput, ContrastiveSearchOutput]

logger = logging.getLogger(__name__)


####################################################
# This is a conversion method from TF 1.0 to PyTorch
# More details: https://medium.com/huggingface/from-tensorflow-to-pytorch-265f40ef2a28
####################################################
def load_tf_weights_in_t5(model, config, tf_checkpoint_path):
    """ Load tf checkpoints in a pytorch model.
    """
    try:
        import re
        import numpy as np
        import tensorflow as tf
    except ImportError:
        logger.error(
            "Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise
    tf_path = os.path.abspath(tf_checkpoint_path)
    logger.info("Converting TensorFlow checkpoint from {}".format(tf_path))
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    tf_weights = {}
    for name, shape in init_vars:
        logger.info("Loading TF weight {} with shape {}".format(name, shape))
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        tf_weights[name] = array

    for txt_name in names:
        name = txt_name.split("/")
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if any(
                n in ["adam_v", "adam_m", "AdamWeightDecayOptimizer", "AdamWeightDecayOptimizer_1", "global_step"]
                for n in name
        ):
            logger.info("Skipping {}".format("/".join(name)))
            tf_weights.pop(txt_name, None)
            continue
        if "_slot_" in name[-1]:
            logger.info("Skipping {}".format("/".join(name)))
            tf_weights.pop(txt_name, None)
            continue
        pointer = model
        array = tf_weights[txt_name]
        for m_name in name:
            if re.fullmatch(r"[A-Za-z]+_\d+", m_name):
                scope_names = re.split(r"_(\d+)", m_name)
            else:
                scope_names = [m_name]
            if scope_names[0] in ["kernel", "scale", "embedding"]:
                pointer = getattr(pointer, "weight")
            # elif scope_names[0] == 'scale':
            #     pointer = getattr(pointer, 'weight')
            # elif scope_names[0] == 'output_bias' or scope_names[0] == 'beta':
            #     pointer = getattr(pointer, 'bias')
            # elif scope_names[0] == 'squad':
            #     pointer = getattr(pointer, 'classifier')
            else:
                try:
                    pointer = getattr(pointer, scope_names[0])
                except AttributeError:
                    logger.info("Skipping {}".format("/".join(name)))
                    continue
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]
        if scope_names[0] not in ["kernel", "scale", "embedding"]:
            pointer = getattr(pointer, "weight")
        if scope_names[0] != "embedding":
            logger.info("Transposing numpy weight of shape {} for {}".format(array.shape, name))
            array = np.transpose(array)
        try:
            assert pointer.shape == array.shape
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        logger.info("Initialize PyTorch weight {}".format(name))
        pointer.data = torch.from_numpy(array.astype(np.float32))
        tf_weights.pop(txt_name, None)

    logger.info("Weights not copied to PyTorch model: {}".format(", ".join(tf_weights.keys())))
    # logger.info("Weights not copied to PyTorch model: {}".format(', '.join(tf_weights.keys())))
    return model


####################################################
# PyTorch Models are constructed by sub-classing
# - torch.nn.Module for the layers and
# - PreTrainedModel for the models (it-self a sub-class of nn.Module)
####################################################
PARALLELIZE_DOCSTRING = r"""
    This is an experimental feature and is a subject to change at a moment's notice.

    Uses a device map to distribute attention modules of the model across several devices. If no device map is given,
    it will evenly distribute blocks across all devices.

    Args:
        device_map (`Dict[int, list]`, optional, defaults to None):
            A dictionary that maps attention modules to devices. Note that the embedding module and LMHead are always
            automatically mapped to the first device (for esoteric reasons). That means that the first device should
            have fewer attention modules mapped to it than other devices. For reference, the t5 models have the
            following number of attention modules:

                - t5-small: 6
                - t5-base: 12
                - t5-large: 24
                - t5-3b: 24
                - t5-11b: 24

    Example:

    ```python
    # Here is an example of a device map on a machine with 4 GPUs using t5-3b, which has a total of 24 attention modules:
    model = T5ForConditionalGeneration.from_pretrained("t5-3b")
    device_map = {
        0: [0, 1, 2],
        1: [3, 4, 5, 6, 7, 8, 9],
        2: [10, 11, 12, 13, 14, 15, 16],
        3: [17, 18, 19, 20, 21, 22, 23],
    }
    model.parallelize(device_map)
    ```
"""
DEPARALLELIZE_DOCSTRING = r"""
    Moves the model to cpu from a model parallel state.

    Example:

    ```python
    # On a 4 GPU machine with t5-3b:
    model = T5ForConditionalGeneration.from_pretrained("t5-3b")
    device_map = {
        0: [0, 1, 2],
        1: [3, 4, 5, 6, 7, 8, 9],
        2: [10, 11, 12, 13, 14, 15, 16],
        3: [17, 18, 19, 20, 21, 22, 23],
    }
    model.parallelize(device_map)  # Splits the model across several devices
    model.deparallelize()  # Put the model back on cpu and cleans memory by calling torch.cuda.empty_cache()
    ```
"""


def invert_mask(attention_mask):
    """Turns 1->0, 0->1, False->True, True-> False"""
    assert attention_mask.dim() == 2
    return attention_mask.eq(0)


class T5LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """ Construct a layernorm module in the T5 style
            No bias and no substraction of mean.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        # T5 uses a layer_norm which only scales and doesn't shift, which is also known as Root Mean
        # Square Layer Normalization https://arxiv.org/abs/1910.07467 thus varience is calculated
        # w/o mean and there is no bias. Additionally we want to make sure that the accumulation for
        # half-precision inputs is done in fp32
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        x = x / torch.sqrt(variance + self.variance_epsilon)

        if self.weight.dtype == torch.float16:
            x = x.to(torch.float16)
        return self.weight * x


class T5DenseReluDense(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.wi = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states):
        h = self.wi(hidden_states)
        h = F.relu(h)
        h = self.dropout(h)
        h = self.wo(h)
        return h


class T5LayerFF(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.DenseReluDense = T5DenseReluDense(config)
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states):
        norm_x = self.layer_norm(hidden_states)
        y = self.DenseReluDense(norm_x)
        layer_output = hidden_states + self.dropout(y)
        return layer_output


class T5Attention(nn.Module):
    def __init__(self, config: T5Config, has_relative_attention_bias=False):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.has_relative_attention_bias = has_relative_attention_bias
        self.relative_attention_max_distance = config.relative_attention_max_distance
        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        self.d_model = config.d_model
        self.d_kv = config.d_kv
        self.n_heads = config.num_heads
        self.dropout = config.dropout_rate
        self.inner_dim = self.n_heads * self.d_kv

        # Mesh TensorFlow initialization to avoid scaling before softmax
        self.q = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.k = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.v = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.o = nn.Linear(self.inner_dim, self.d_model, bias=False)

        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(self.relative_attention_num_buckets, self.n_heads)
        self.pruned_heads = set()
        self.gradient_checkpointing = False

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(heads, self.n_heads, self.d_kv, self.pruned_heads)
        # Prune linear layers
        self.q = prune_linear_layer(self.q, index)
        self.k = prune_linear_layer(self.k, index)
        self.v = prune_linear_layer(self.v, index)
        self.o = prune_linear_layer(self.o, index, dim=1)
        # Update hyper params
        self.n_heads = self.n_heads - len(heads)
        self.inner_dim = self.d_kv * self.n_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593

        Translate relative position to a bucket number for relative attention.
        The relative position is defined as memory_position - query_position, i.e.
        the distance in tokens from the attending position to the attended-to
        position.  If bidirectional=False, then positive relative positions are
        invalid.
        We use smaller buckets for small absolute relative_position and larger buckets
        for larger absolute relative_positions.  All relative positions >=max_distance
        map to the same bucket.  All relative positions <=-max_distance map to the
        same bucket.  This should allow for more graceful generalization to longer
        sequences than the model has been trained on.
        Args:
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer
        Returns:
            a Tensor with the same shape as relative_position, containing int32
            values in the range [0, num_buckets)
        """
        ret = 0
        n = -relative_position
        if bidirectional:
            num_buckets //= 2
            ret += (n < 0).to(torch.long) * num_buckets  # mtf.to_int32(mtf.less(n, 0)) * num_buckets
            n = torch.abs(n)
        else:
            n = torch.max(n, torch.zeros_like(n))
        # now n is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = n < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        val_if_large = max_exact + (
                torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).to(torch.long)
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def compute_bias(self, qlen, klen):
        """ Compute binned relative position bias """
        context_position = torch.arange(qlen, dtype=torch.long)[:, None]
        memory_position = torch.arange(klen, dtype=torch.long)[None, :]
        relative_position = memory_position - context_position  # shape (qlen, klen)
        rp_bucket = self._relative_position_bucket(
            relative_position,  # shape (qlen, klen)
            bidirectional=not self.is_decoder,
            num_buckets=self.relative_attention_num_buckets,
        )
        rp_bucket = rp_bucket.to(self.relative_attention_bias.weight.device)
        values = self.relative_attention_bias(rp_bucket)  # shape (qlen, klen, num_heads)
        values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, qlen, klen)
        return values

    def forward(
            self,
            input,
            mask=None,
            kv=None,
            position_bias=None,
            past_key_values=None,
            head_mask=None,
            query_length=None,
            use_cache=False,
            output_attentions=False,
    ):
        """
        Self-attention (if kv is None) or attention over source sentence (provided by kv).
        """
        # Input is (bs, qlen, dim)
        # Mask is (bs, klen) (non-causal) or (bs, klen, klen)
        # past_key_values[0] is (bs, n_heads, q_len - 1, dim_per_head)
        bs, qlen, dim = input.size()

        if past_key_values is not None:
            assert self.is_decoder is True, "Encoder cannot cache past key value states"
            assert (
                    len(past_key_values) == 2
            ), "past_key_values should have 2 past states: keys and values. Got {} past states".format(
                len(past_key_values)
            )
            real_qlen = qlen + past_key_values[0].shape[2] if query_length is None else query_length
        else:
            real_qlen = qlen

        if kv is None:
            klen = real_qlen
        else:
            klen = kv.size(1)

        def shape(x):
            """  projection """
            return x.view(bs, -1, self.n_heads, self.d_kv).transpose(1, 2)

        def unshape(x):
            """  compute context """
            return x.transpose(1, 2).contiguous().view(bs, -1, self.inner_dim)

        q = shape(self.q(input))  # (bs, n_heads, qlen, dim_per_head)

        if kv is None:
            k = shape(self.k(input))  # (bs, n_heads, qlen, dim_per_head)
            v = shape(self.v(input))  # (bs, n_heads, qlen, dim_per_head)
        elif past_key_values is None:
            k = v = kv
            k = shape(self.k(k))  # (bs, n_heads, qlen, dim_per_head)
            v = shape(self.v(v))  # (bs, n_heads, qlen, dim_per_head)

        if past_key_values is not None:
            if kv is None:
                k_, v_ = past_key_values
                k = torch.cat([k_, k], dim=2)  # (bs, n_heads, klen, dim_per_head)
                v = torch.cat([v_, v], dim=2)  # (bs, n_heads, klen, dim_per_head)
            else:
                k, v = past_key_values

        if self.is_decoder and use_cache is True:
            present_key_value_state = ((k, v),)
        else:
            present_key_value_state = (None,)

        scores = torch.einsum("bnqd,bnkd->bnqk", q, k)  # (bs, n_heads, qlen, klen)

        if position_bias is None:
            if not self.has_relative_attention_bias:
                raise ValueError("No position_bias provided and no weights to compute position_bias")
            position_bias = self.compute_bias(real_qlen, klen)

            # if key and values are already calculated
            # we want only the last query position bias
            if past_key_values is not None:
                position_bias = position_bias[:, :, -1:, :]

            if mask is not None:
                position_bias = position_bias + mask  # (bs, n_heads, qlen, klen)

        scores += position_bias
        weights = F.softmax(scores.float(), dim=-1).type_as(scores)  # (bs, n_heads, qlen, klen)
        weights = F.dropout(weights, p=self.dropout, training=self.training)  # (bs, n_heads, qlen, klen)

        # Mask heads if we want to
        if head_mask is not None:
            weights = weights * head_mask

        context = torch.matmul(weights, v)  # (bs, n_heads, qlen, dim_per_head)
        context = unshape(context)  # (bs, qlen, dim)

        context = self.o(context)

        outputs = (context,) + present_key_value_state
        # print(output_attentions, )
        if output_attentions:
            outputs = outputs + (weights,)
        if self.has_relative_attention_bias:
            outputs = outputs + (position_bias,)
        return outputs


# Structure-aware self-attention
class T5StructAttention(nn.Module):
    def __init__(self, config: T5Config, has_relative_attention_bias=False):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.has_relative_attention_bias = has_relative_attention_bias

        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        self.d_model = config.d_model
        self.d_kv = config.d_kv
        self.n_heads = config.num_heads
        self.dropout = config.dropout_rate
        self.inner_dim = self.n_heads * self.d_kv

        # Mesh TensorFlow initialization to avoid scaling before softmax
        self.q = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.k = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.v = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.rel_k = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.rel_v = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.o = nn.Linear(self.inner_dim, self.d_model, bias=False)

        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(self.relative_attention_num_buckets, self.n_heads)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(heads, self.n_heads, self.d_kv, self.pruned_heads)
        # Prune linear layers
        self.q = prune_linear_layer(self.q, index)
        self.k = prune_linear_layer(self.k, index)
        self.v = prune_linear_layer(self.v, index)
        self.rel_k = prune_linear_layer(self.k_rel, index)
        self.rel_v = prune_linear_layer(self.v_rel, index)
        self.o = prune_linear_layer(self.o, index, dim=1)
        # Update hyper params
        self.n_heads = self.n_heads - len(heads)
        self.inner_dim = self.d_kv * self.n_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593

        Translate relative position to a bucket number for relative attention.
        The relative position is defined as memory_position - query_position, i.e.
        the distance in tokens from the attending position to the attended-to
        position.  If bidirectional=False, then positive relative positions are
        invalid.
        We use smaller buckets for small absolute relative_position and larger buckets
        for larger absolute relative_positions.  All relative positions >=max_distance
        map to the same bucket.  All relative positions <=-max_distance map to the
        same bucket.  This should allow for more graceful generalization to longer
        sequences than the model has been trained on.
        Args:
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer
        Returns:
            a Tensor with the same shape as relative_position, containing int32
            values in the range [0, num_buckets)
        """
        ret = 0
        n = -relative_position
        if bidirectional:
            num_buckets //= 2
            ret += (n < 0).to(torch.long) * num_buckets  # mtf.to_int32(mtf.less(n, 0)) * num_buckets
            n = torch.abs(n)
        else:
            n = torch.max(n, torch.zeros_like(n))
        # now n is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = n < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        val_if_large = max_exact + (
                torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).to(torch.long)
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def compute_bias(self, qlen, klen):
        """ Compute binned relative position bias """
        context_position = torch.arange(qlen, dtype=torch.long)[:, None]
        memory_position = torch.arange(klen, dtype=torch.long)[None, :]
        relative_position = memory_position - context_position  # shape (qlen, klen)
        rp_bucket = self._relative_position_bucket(
            relative_position,  # shape (qlen, klen)
            bidirectional=not self.is_decoder,
            num_buckets=self.relative_attention_num_buckets,
        )
        rp_bucket = rp_bucket.to(self.relative_attention_bias.weight.device)
        values = self.relative_attention_bias(rp_bucket)  # shape (qlen, klen, num_heads)
        values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, qlen, klen)
        return values

    def forward(
            self,
            input,
            mask=None,
            kv=None,
            position_bias=None,
            past_key_values=None,
            head_mask=None,
            query_length=None,
            use_cache=False,
            output_attentions=False,
            adj_matrix_emb=None,
    ):
        """
        Self-attention (if kv is None) or attention over source sentence (provided by kv).
        """
        # Input is (bs, qlen, dim)
        # Mask is (bs, klen) (non-causal) or (bs, klen, klen)
        # past_key_values[0] is (bs, n_heads, q_len - 1, dim_per_head)
        bs, qlen, dim = input.size()

        if past_key_values is not None:
            assert self.is_decoder is True, "Encoder cannot cache past key value states"
            assert (
                    len(past_key_values) == 2
            ), "past_key_values should have 2 past states: keys and values. Got {} past states".format(
                len(past_key_values)
            )
            real_qlen = qlen + past_key_values[0].shape[2] if query_length is None else query_length
        else:
            real_qlen = qlen

        if kv is None:
            klen = real_qlen
        else:
            klen = kv.size(1)

        def shape(x):
            """  projection """
            return x.view(bs, -1, self.n_heads, self.d_kv).transpose(1, 2)

        def unshape(x):
            """  compute context """
            return x.transpose(1, 2).contiguous().view(bs, -1, self.inner_dim)

        q = shape(self.q(input))  # (bs, n_heads, qlen, dim_per_head)

        if kv is None:
            k = shape(self.k(input))  # bs * n_heads * klen * dim_per_head
            v = shape(self.v(input))  # bs * n_heads * klen * dim_per_head
        elif past_key_values is None:
            k = v = kv
            k = shape(self.k(k))  # bs * n_heads * klen * dim_per_head
            v = shape(self.v(v))  # bs * n_heads * klen * dim_per_head

        rel_k = shape(self.rel_k(adj_matrix_emb)).contiguous().view(bs, self.n_heads, qlen, -1,
                                                                    self.d_kv)  # bs * n_heads * qlen * klen * dim_per_head
        rel_v = shape(self.rel_v(adj_matrix_emb)).contiguous().view(bs, self.n_heads, qlen, -1,
                                                                    self.d_kv)  # bs * n_heads * qlen * klen * dim_per_head)

        if past_key_values is not None:
            if kv is None:
                k_, v_ = past_key_values
                k = torch.cat([k_, k], dim=2)  # bs * n_heads * klen * dim_per_head
                v = torch.cat([v_, v], dim=2)  # bs * n_heads * klen * dim_per_head
            else:
                k, v = past_key_values

        if self.is_decoder and use_cache is True:
            present_key_value_state = ((k, v),)
        else:
            present_key_value_state = (None,)

        scores = torch.einsum("bnqd,bnkd->bnqk", q, k)  # bs * n_heads * qlen * klen)
        scores_rel = torch.einsum('aebc,aebdc->aebd', q, rel_k)  # bs * n_heads * qlen * klen
        scores = scores + scores_rel
        scores += mask
        weights = F.softmax(scores.float(), dim=-1).type_as(scores)  # bs * n_heads * qlen * klen
        weights = F.dropout(weights, p=self.dropout, training=self.training)  # bs * n_heads * qlen * klen

        # Mask heads if we want to
        if head_mask is not None:
            weights = weights * head_mask

        context = torch.matmul(weights, v)  # bs * n_heads * qlen * dim_per_head

        context_rel = torch.einsum('abcd,abcde->abce', weights, rel_v)  # bs * n_heads * qlen * dim_per_head

        context = context + context_rel

        context = unshape(context)  # bs * qlen * dim

        context = self.o(context)

        outputs = (context,) + present_key_value_state

        if output_attentions:
            outputs = outputs + (weights,)
        if self.has_relative_attention_bias:
            outputs = outputs + (position_bias,)
        return outputs


class T5LayerSelfAttention(nn.Module):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__()
        self.SelfAttention = T5Attention(config, has_relative_attention_bias=has_relative_attention_bias)
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            position_bias=None,
            head_mask=None,
            past_key_values=None,
            use_cache=False,
            output_attentions=False,
    ):
        norm_x = self.layer_norm(hidden_states)
        attention_output = self.SelfAttention(
            norm_x,
            mask=attention_mask,
            position_bias=position_bias,
            head_mask=head_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        y = attention_output[0]
        layer_output = hidden_states + self.dropout(y)
        outputs = (layer_output,) + attention_output[1:]  # add attentions if we output them
        return outputs


# Structure-aware semantic aggregation module
class T5StructLayerSelfAttention(nn.Module):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__()
        self.SelfAttention = T5Attention(config, has_relative_attention_bias=has_relative_attention_bias)
        self.SelfStructAttention = T5StructAttention(config, has_relative_attention_bias=has_relative_attention_bias)
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            input_node_ids=None,
            input_edge_ids=None,
            node_attention_mask=None,
            edge_attention_mask=None,
            extended_node_attention_mask=None,
            extended_edge_attention_mask=None,
            adj_matrix=None,
            position_bias=None,
            head_mask=None,
            past_key_values=None,
            use_cache=False,
            output_attentions=False,
    ):
        norm_x = self.layer_norm(hidden_states)
        attention_output = self.SelfAttention(
            norm_x,
            mask=attention_mask,
            position_bias=position_bias,
            head_mask=head_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        y = attention_output[0]

        # get each representation vector of nodes
        x_ori = y  # batch * len * embed_dim
        x_node = scatter_mean(x_ori, input_node_ids, dim=1)  # batch * node_size * embed_dim
        assert x_node.size(1) == node_attention_mask.size(1) == adj_matrix.size(1)

        # get each representation vector of edges
        x_edge = scatter_mean(x_ori, input_edge_ids, dim=1)  # batch * edge_size * embed_dim
        assert x_edge.size(1) == edge_attention_mask.size(1)
        x_edge = edge_attention_mask.unsqueeze(-1).repeat(1, 1, x_edge.size(2)) * x_edge

        # lookup adj_matrix in edge representation
        node_len, edge_len, emb_dim = adj_matrix.size(1), x_edge.size(1), x_edge.size(2)
        adj_matrix_tmp = adj_matrix.contiguous().view(-1, node_len * node_len).unsqueeze(-1).repeat(1, 1, emb_dim)
        adj_matrix_emb = torch.gather(x_edge, 1, adj_matrix_tmp).view(-1, node_len, node_len,
                                                                      emb_dim)  # batch * node_len * node_len emb_dim
        adj_matrix_emb = adj_matrix_emb.contiguous().view(-1, node_len * node_len,
                                                          emb_dim)  # batch * (node_len * node_len) * emb_dim

        struct_attention_output = self.SelfStructAttention(
            x_node, mask=extended_node_attention_mask, position_bias=None,
            head_mask=head_mask, past_key_values=past_key_values, use_cache=use_cache,
            output_attentions=output_attentions, adj_matrix_emb=adj_matrix_emb,
        )
        x_node = struct_attention_output[0]
        x_node = node_attention_mask.unsqueeze(-1).repeat(1, 1, x_node.size(2)) * x_node

        y = y + torch.gather(x_node, 1, input_node_ids.unsqueeze(-1).repeat(1, 1, x_node.size(2)))

        layer_output = hidden_states + self.dropout(y)
        outputs = (layer_output,) + attention_output[1:]  # add attentions if we output them
        return outputs


class T5LayerCrossAttention(nn.Module):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__()
        self.EncDecAttention = T5Attention(config, has_relative_attention_bias=has_relative_attention_bias)
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
            self,
            hidden_states,
            kv,
            attention_mask=None,
            position_bias=None,
            head_mask=None,
            past_key_values=None,
            use_cache=False,
            query_length=None,
            output_attentions=False,
    ):
        norm_x = self.layer_norm(hidden_states)
        attention_output = self.EncDecAttention(
            norm_x,
            mask=attention_mask,
            kv=kv,
            position_bias=position_bias,
            head_mask=head_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            query_length=query_length,
            output_attentions=output_attentions,
        )
        y = attention_output[0]
        layer_output = hidden_states + self.dropout(y)
        outputs = (layer_output,) + attention_output[1:]  # add attentions if we output them
        return outputs


class T5Block(nn.Module):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.layer = nn.ModuleList()
        self.layer.append(T5LayerSelfAttention(config, has_relative_attention_bias=has_relative_attention_bias))
        if self.is_decoder:
            self.layer.append(T5LayerCrossAttention(config, has_relative_attention_bias=has_relative_attention_bias))

        self.layer.append(T5LayerFF(config))

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            position_bias=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            encoder_decoder_position_bias=None,
            head_mask=None,
            cross_attn_layer_head_mask=None,
            past_key_values=None,
            use_cache=False,
            output_attentions=False,
            return_dict=True,
    ):

        if past_key_values is not None:
            assert self.is_decoder, "Only decoder can use `past_key_valuess`"
            expected_num_past_key_valuess = 2 if encoder_hidden_states is None else 4

            error_message = "There should be {} past states. 2 (past / key) for self attention.{} Got {} past key / value states".format(
                expected_num_past_key_valuess,
                "2 (past / key) for cross attention" if expected_num_past_key_valuess == 4 else "",
                len(past_key_values),
            )
            assert len(past_key_values) == expected_num_past_key_valuess, error_message

            self_attn_past_key_values = past_key_values[:2]
            cross_attn_past_key_values = past_key_values[2:]
        else:
            self_attn_past_key_values, cross_attn_past_key_values = None, None

        self_attention_outputs = self.layer[0](
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            head_mask=head_mask,
            past_key_values=self_attn_past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        hidden_states, present_key_value_state = self_attention_outputs[:2]
        attention_outputs = self_attention_outputs[2:]  # Keep self-attention outputs and relative position weights
        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16 and torch.isinf(hidden_states).any():
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        if self.is_decoder and encoder_hidden_states is not None:
            # the actual query length is unknown for cross attention
            # if using past key value states. Need to inject it here
            if present_key_value_state is not None:
                query_length = present_key_value_state[0].shape[2]
            else:
                query_length = None

            cross_attention_outputs = self.layer[1](
                hidden_states,
                kv=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                position_bias=encoder_decoder_position_bias,
                head_mask=cross_attn_layer_head_mask,
                past_key_values=cross_attn_past_key_values,
                query_length=query_length,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            hidden_states = cross_attention_outputs[0]
            # clamp inf values to enable fp16 training
            if hidden_states.dtype == torch.float16 and torch.isinf(hidden_states).any():
                clamp_value = torch.finfo(hidden_states.dtype).max - 1000
                hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
            # Combine self attn and cross attn key value states
            if present_key_value_state is not None:
                present_key_value_state = present_key_value_state + cross_attention_outputs[1]
            # Keep cross-attention outputs and relative position weights
            attention_outputs = attention_outputs + cross_attention_outputs[2:]
        # Apply Feed Forward layer
        hidden_states = self.layer[-1](hidden_states)
        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16 and torch.isinf(hidden_states).any():
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        outputs = (hidden_states,)
        # Add attentions if we output them
        outputs = outputs + (present_key_value_state,) + attention_outputs
        return outputs  # hidden-states, present_key_value_states, (self-attention weights), (self-attention position bias), (cross-attention weights), (cross-attention position bias)


# T5 block with structure-aware semantic aggregation module
class T5StructBlock(nn.Module):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.layer = nn.ModuleList()
        self.layer.append(T5StructLayerSelfAttention(config, has_relative_attention_bias=has_relative_attention_bias))
        if self.is_decoder:
            self.layer.append(T5LayerCrossAttention(config, has_relative_attention_bias=has_relative_attention_bias))

        self.layer.append(T5LayerFF(config))

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            input_node_ids=None,
            input_edge_ids=None,
            node_attention_mask=None,
            edge_attention_mask=None,
            extended_node_attention_mask=None,
            extended_edge_attention_mask=None,
            adj_matrix=None,
            position_bias=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            encoder_decoder_position_bias=None,
            layer_head_mask=None,
            cross_attn_layer_head_mask=None,
            past_key_values=None,
            use_cache=False,
            output_attentions=False,
            return_dict=True,
    ):

        if past_key_values is not None:
            assert self.is_decoder, "Only decoder can use `past_key_valuess`"
            expected_num_past_key_values = 2 if encoder_hidden_states is None else 4

            if len(past_key_values) != expected_num_past_key_values:
                raise ValueError(
                    f"There should be {expected_num_past_key_values} past states. "
                    f"{'2 (past / key) for cross attention. ' if expected_num_past_key_values == 4 else ''}"
                    f"Got {len(past_key_values)} past key / value states")

            self_attn_past_key_values = past_key_values[:2]
            cross_attn_past_key_values = past_key_values[2:]
        else:
            self_attn_past_key_values, cross_attn_past_key_values = None, None

        self_attention_outputs = self.layer[0](
            hidden_states,
            attention_mask=attention_mask,
            input_node_ids=input_node_ids,
            input_edge_ids=input_edge_ids,
            node_attention_mask=node_attention_mask,
            edge_attention_mask=edge_attention_mask,
            extended_node_attention_mask=extended_node_attention_mask,
            extended_edge_attention_mask=extended_edge_attention_mask,
            adj_matrix=adj_matrix,
            position_bias=position_bias,
            head_mask=layer_head_mask,
            past_key_values=self_attn_past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions, )
        hidden_states, present_key_value_state = self_attention_outputs[:2]
        attention_outputs = self_attention_outputs[2:]  # Keep self-attention outputs and relative position weights

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16 and torch.isinf(hidden_states).any():
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        if self.is_decoder and encoder_hidden_states is not None:
            # the actual query length is unknown for cross attention
            # if using past key value states. Need to inject it here
            if present_key_value_state is not None:
                query_length = present_key_value_state[0].shape[2]
            else:
                query_length = None

            cross_attention_outputs = self.layer[1](
                hidden_states,
                kv=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                position_bias=encoder_decoder_position_bias,
                head_mask=cross_attn_layer_head_mask,
                past_key_values=cross_attn_past_key_values,
                query_length=query_length,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            hidden_states = cross_attention_outputs[0]

            # clamp inf values to enable fp16 training
            if hidden_states.dtype == torch.float16 and torch.isinf(hidden_states).any():
                clamp_value = torch.finfo(hidden_states.dtype).max - 1000
                hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

            # Combine self attn and cross attn key value states
            if present_key_value_state is not None:
                present_key_value_state = present_key_value_state + cross_attention_outputs[1]

            # Keep cross-attention outputs and relative position weights
            attention_outputs = attention_outputs + cross_attention_outputs[2:]

        # Apply Feed Forward layer
        hidden_states = self.layer[-1](hidden_states)

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16 and torch.isinf(hidden_states).any():
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        # Add attentions if we output them
        if use_cache:
            outputs = outputs + (present_key_value_state,) + attention_outputs
        else:
            outputs = outputs + attention_outputs
        # hidden-states, present_key_value_states, (self-attention weights), (self-attention position bias),
        # (cross-attention weights), (cross-attention position bias)

        return outputs


class T5PreTrainedModel(PreTrainedModel):
    """ An abstract class to handle weights initialization and
        a simple interface for downloading and loading pretrained models.
    """

    config_class = T5Config
    load_tf_weights = load_tf_weights_in_t5
    base_model_prefix = "transformer"
    is_parallelizable = True
    supports_gradient_checkpointing = True
    _no_split_modules = ["T5Block"]

    @property
    def dummy_inputs(self):
        input_ids = torch.tensor(DUMMY_INPUTS)
        input_mask = torch.tensor(DUMMY_MASK)
        dummy_inputs = {
            "decoder_input_ids": input_ids,
            "input_ids": input_ids,
            "decoder_attention_mask": input_mask,
        }
        return dummy_inputs

    def _init_weights(self, module):
        """ Initialize the weights """
        factor = self.config.initializer_factor  # Used for testing weights initialization
        if isinstance(module, T5LayerNorm):
            module.weight.data.fill_(factor * 1.0)
        elif isinstance(module, (T5Model, T5ForConditionalGeneration)):
            # Mesh TensorFlow embeddings initialization
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L1624
            module.shared.weight.data.normal_(mean=0.0, std=factor * 1.0)
            if hasattr(module, "lm_head") and not self.config.tie_word_embeddings:
                module.lm_head.weight.data.normal_(mean=0.0, std=factor * 1.0)
        elif isinstance(module, T5DenseReluDense):
            # Mesh TensorFlow FF initialization
            # See https://github.com/tensorflow/mesh/blob/master/mesh_tensorflow/transformer/transformer_layers.py#L56
            # and https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L89
            module.wi.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_model) ** -0.5))
            if hasattr(module.wi, "bias") and module.wi.bias is not None:
                module.wi.bias.data.zero_()
            module.wo.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_ff) ** -0.5))
            if hasattr(module.wo, "bias") and module.wo.bias is not None:
                module.wo.bias.data.zero_()
        elif isinstance(module, T5Attention):
            # Mesh TensorFlow attention initialization to avoid scaling before softmax
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/attention.py#L136
            d_model = self.config.d_model
            d_kv = self.config.d_kv
            n_heads = self.config.num_heads
            module.q.weight.data.normal_(mean=0.0, std=factor * ((d_model * d_kv) ** -0.5))
            module.k.weight.data.normal_(mean=0.0, std=factor * (d_model ** -0.5))
            module.v.weight.data.normal_(mean=0.0, std=factor * (d_model ** -0.5))
            module.o.weight.data.normal_(mean=0.0, std=factor * ((n_heads * d_kv) ** -0.5))
            if module.has_relative_attention_bias:
                module.relative_attention_bias.weight.data.normal_(mean=0.0, std=factor * ((d_model) ** -0.5))

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (T5Attention, T5Stack)):
            module.gradient_checkpointing = value

    def _shift_right(self, input_ids):
        decoder_start_token_id = self.config.decoder_start_token_id
        pad_token_id = self.config.pad_token_id

        assert (
                decoder_start_token_id is not None
        ), "self.model.config.decoder_start_token_id has to be defined. In T5 it is usually set to the pad_token_id. " \
           "See T5 docs for more information "

        # shift inputs to the right
        if is_torch_fx_proxy(input_ids):
            # Item assignment is not supported natively for proxies.
            shifted_input_ids = torch.full(input_ids.shape[:-1] + (1,), decoder_start_token_id)
            shifted_input_ids = torch.cat([shifted_input_ids, input_ids[..., :-1]], dim=-1)
        else:
            shifted_input_ids = input_ids.new_zeros(input_ids.shape)
            shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
            shifted_input_ids[..., 0] = decoder_start_token_id

        assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        assert torch.all(shifted_input_ids >= 0).item(), "Verify that `labels` has only positive values and -100"

        return shifted_input_ids


class T5Stack(T5PreTrainedModel):
    def __init__(self, config, embed_tokens=None):
        super().__init__(config)

        self.embed_tokens = embed_tokens
        self.is_decoder = config.is_decoder

        self.block = nn.ModuleList(
            [T5Block(config, has_relative_attention_bias=bool(i == 0)) for i in range(config.num_layers)]
        )
        self.final_layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

        # Initialize weights and apply final processing
        self.post_init()
        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = False

    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=None):
        # Check validity of device_map
        self.device_map = (
            get_device_map(len(self.block), range(torch.cuda.device_count())) if device_map is None else device_map
        )
        assert_device_map(self.device_map, len(self.block))
        self.model_parallel = True
        self.first_device = "cpu" if "cpu" in self.device_map.keys() else "cuda:" + str(min(self.device_map.keys()))
        self.last_device = "cuda:" + str(max(self.device_map.keys()))
        # Load onto devices
        for k, v in self.device_map.items():
            for layer in v:
                cuda_device = "cuda:" + str(k)
                self.block[layer] = self.block[layer].to(cuda_device)

        # Set embed_tokens to first layer
        self.embed_tokens = self.embed_tokens.to(self.first_device)
        # Set final layer norm to last device
        self.final_layer_norm = self.final_layer_norm.to(self.last_device)

    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def deparallelize(self):
        self.model_parallel = False
        self.device_map = None
        self.first_device = "cpu"
        self.last_device = "cpu"
        for i in range(len(self.block)):
            self.block[i] = self.block[i].to("cpu")
        self.embed_tokens = self.embed_tokens.to("cpu")
        self.final_layer_norm = self.final_layer_norm.to("cpu")
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.embed_tokens

    def get_output_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, new_embeddings):
        self.embed_tokens = new_embeddings

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            inputs_embeds=None,
            head_mask=None,
            cross_attn_head_mask=None,
            past_key_values=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        # Model parallel
        if self.model_parallel:
            torch.cuda.set_device(self.first_device)
            self.embed_tokens = self.embed_tokens.to(self.first_device)
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            if self.is_decoder:
                raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")
            else:
                raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            assert self.embed_tokens is not None, "You have to initialise the model with valid token embeddings"
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = input_shape
        # required mask seq length can be calculated via length of past
        mask_seq_length = past_key_values[0][0].shape[2] + seq_length if past_key_values is not None else seq_length

        if attention_mask is None:
            attention_mask = torch.ones(batch_size, mask_seq_length).to(inputs_embeds.device)
        if self.is_decoder and encoder_attention_mask is None and encoder_hidden_states is not None:
            encoder_seq_length = encoder_hidden_states.shape[1]
            encoder_attention_mask = torch.ones(
                batch_size, encoder_seq_length, device=inputs_embeds.device, dtype=torch.long)

        # initialize past_key_valuess with `None` if past does not exist
        if past_key_values is None:
            past_key_values = [None] * len(self.block)

        # ourselves in which case we just need to make it broadcastable to all heads.
        # after this, zero means attending while -10000.0 means no attending
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, inputs_embeds.device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.is_decoder and encoder_attention_mask is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=inputs_embeds.device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)
        cross_attn_head_mask = self.get_head_mask(cross_attn_head_mask, self.config.num_layers)
        present_key_value_states = ()
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and self.is_decoder) else None
        position_bias = None
        encoder_decoder_position_bias = None

        hidden_states = self.dropout(inputs_embeds)

        for i, (layer_module, past_key_values) in enumerate(zip(self.block, past_key_values)):
            layer_head_mask = head_mask[i]
            cross_attn_layer_head_mask = cross_attn_head_mask[i]
            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if position_bias is not None:
                    position_bias = position_bias.to(hidden_states.device)
                if encoder_hidden_states is not None:
                    encoder_hidden_states = encoder_hidden_states.to(hidden_states.device)
                if encoder_extended_attention_mask is not None:
                    encoder_extended_attention_mask = encoder_extended_attention_mask.to(hidden_states.device)
                if encoder_decoder_position_bias is not None:
                    encoder_decoder_position_bias = encoder_decoder_position_bias.to(hidden_states.device)
                if layer_head_mask is not None:
                    layer_head_mask = layer_head_mask.to(hidden_states.device)
                if cross_attn_layer_head_mask is not None:
                    cross_attn_layer_head_mask = cross_attn_layer_head_mask.to(hidden_states.device)

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with gradient checkpointing. "
                        "Setting `use_cache=False`...")
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return tuple(module(*inputs, use_cache, output_attentions))

                    return custom_forward

                layer_outputs = checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    extended_attention_mask,
                    position_bias,
                    encoder_hidden_states,
                    encoder_extended_attention_mask,
                    encoder_decoder_position_bias,
                    layer_head_mask,
                    cross_attn_layer_head_mask,
                    None,  # past_key_value is always None with gradient checkpointing
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask=extended_attention_mask,
                    position_bias=position_bias,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_extended_attention_mask,
                    encoder_decoder_position_bias=encoder_decoder_position_bias,
                    head_mask=layer_head_mask,
                    cross_attn_layer_head_mask=cross_attn_layer_head_mask,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    output_attentions=output_attentions, )
            # layer_outputs is a tuple with:
            # hidden-states, key-value-states, (self-attention weights),
            # (self-attention position bias), (cross-attention weights),
            # (cross-attention position bias)
            if use_cache is False:
                layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]
            hidden_states, present_key_value_state = layer_outputs[:2]

            # We share the position biases between the layers - the first layer store them
            # layer_outputs = hidden-states, key-value-states
            # (self-attention weights), (self-attention position bias),
            # (cross-attention weights), (cross-attention position bias)
            #         if output_attentions:     copied from attention code
            #             outputs = outputs + (weights,)
            #         if self.has_relative_attention_bias:
            #             outputs = outputs + (position_bias,),
            if i == 0:
                position_bias = layer_outputs[3 if output_attentions else 2]
                if self.is_decoder and encoder_hidden_states is not None:
                    encoder_decoder_position_bias = layer_outputs[5 if output_attentions else 3]
            # append next layer key value states
            present_key_value_states = present_key_value_states + (present_key_value_state,)

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[2],)  # We keep only self-attention weights for now
                if self.is_decoder:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[4],)
            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        # print(return_dict)
        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    present_key_value_states,
                    all_hidden_states,
                    all_attentions,
                    all_cross_attentions, ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=present_key_value_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions, )
        # last-layer hidden state, (presents,) (all hidden states), (all attentions)


def convert_length_to_mask(len_list, max_length):
    len_size = max_length
    batch_size = len_list.size(0)
    seq_range = torch.arange(0, len_size, device=len_list.device)
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, len_size)
    length_expand = len_list.expand_as(seq_range_expand)
    attention_mask = seq_range_expand < length_expand
    return attention_mask


# T5 stack with structure-aware semantic aggregation module
class T5StructStack(T5PreTrainedModel):
    def __init__(self, config, embed_tokens=None):
        super().__init__(config)

        self.embed_tokens = embed_tokens
        self.is_decoder = config.is_decoder

        self.block = nn.ModuleList(
            [T5StructBlock(config, has_relative_attention_bias=bool(i == 0))
             for i in range(config.num_layers)])
        self.final_layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

        # Initialize weights and apply final processing
        self.post_init()
        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = False

    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=None):
        # Check validity of device_map
        self.device_map = (get_device_map(len(self.block), range(torch.cuda.device_count()))
                           if device_map is None else device_map)
        assert_device_map(self.device_map, len(self.block))
        self.model_parallel = True
        self.first_device = "cpu" if "cpu" in self.device_map.keys() else "cuda:" + str(min(self.device_map.keys()))
        self.last_device = "cuda:" + str(max(self.device_map.keys()))
        # Load onto devices
        for k, v in self.device_map.items():
            for layer in v:
                cuda_device = "cuda:" + str(k)
                self.block[layer] = self.block[layer].to(cuda_device)
        # Set embed_tokens to first layer
        self.embed_tokens = self.embed_tokens.to(self.first_device)
        # Set final layer norm to last device
        self.final_layer_norm = self.final_layer_norm.to(self.last_device)

    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def deparallelize(self):
        self.model_parallel = False
        self.device_map = None
        self.first_device = "cpu"
        self.last_device = "cpu"
        for i in range(len(self.block)):
            self.block[i] = self.block[i].to("cpu")
        self.embed_tokens = self.embed_tokens.to("cpu")
        self.final_layer_norm = self.final_layer_norm.to("cpu")
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.embed_tokens

    def get_output_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, new_embeddings):
        self.embed_tokens = new_embeddings

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            input_node_ids=None,
            input_edge_ids=None,
            node_length=None,
            edge_length=None,
            adj_matrix=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            inputs_embeds=None,
            head_mask=None,
            cross_attn_head_mask=None,
            past_key_values=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        # Model parallel
        if self.model_parallel:
            torch.cuda.set_device(self.first_device)
            self.embed_tokens = self.embed_tokens.to(self.first_device)
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(f"You have to specify either {err_msg_prefix}input_ids or {err_msg_prefix}inputs_embeds")

        if inputs_embeds is None:
            assert self.embed_tokens is not None, "You have to intialize the model with valid token embeddings"
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = input_shape

        # required mask seq length can be calculated via length of past
        mask_seq_length = past_key_values[0][0].shape[2] + seq_length if past_key_values is not None else seq_length
        if use_cache is True:
            assert self.is_decoder, f"`use_cache` can only be set to `True` if {self} is used as a decoder"

        if attention_mask is None:
            attention_mask = torch.ones(batch_size, mask_seq_length).to(inputs_embeds.device)
        if self.is_decoder and encoder_attention_mask is None and encoder_hidden_states is not None:
            encoder_seq_length = encoder_hidden_states.shape[1]
            encoder_attention_mask = torch.ones(
                batch_size, encoder_seq_length, device=inputs_embeds.device, dtype=torch.long)

        # initialize past_key_valuess with `None` if past does not exist
        if past_key_values is None:
            past_key_values = [None] * len(self.block)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        # after this, zero means attending while -10000.0 means no attending
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, inputs_embeds.device)

        # build attention mask and invert
        # node/edge_attention_mask: 1 means attending while 0 means no attending (batch, len)
        # extended_node/edge_attention_mask: 0 means attending while -10000.0 means no attending (batch, 1, 1, len)
        node_attention_mask = convert_length_to_mask(node_length, input_node_ids.max() + 1)
        edge_attention_mask = convert_length_to_mask(edge_length, input_edge_ids.max() + 1)
        extended_node_attention_mask = self.get_extended_attention_mask(node_attention_mask,
                                                                        (batch_size, input_node_ids.max() + 1),
                                                                        input_node_ids.device)
        extended_edge_attention_mask = self.get_extended_attention_mask(edge_attention_mask,
                                                                        (batch_size, input_edge_ids.max() + 1),
                                                                        input_edge_ids.device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.is_decoder and encoder_attention_mask is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=inputs_embeds.device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)
        cross_attn_head_mask = self.get_head_mask(cross_attn_head_mask, self.config.num_layers)
        present_key_value_states = ()
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and self.is_decoder) else None
        position_bias = None
        encoder_decoder_position_bias = None

        hidden_states = self.dropout(inputs_embeds)

        for i, (layer_module, past_key_values) in enumerate(zip(self.block, past_key_values)):
            layer_head_mask = head_mask[i]
            cross_attn_layer_head_mask = cross_attn_head_mask[i]
            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if position_bias is not None:
                    position_bias = position_bias.to(hidden_states.device)
                if encoder_hidden_states is not None:
                    encoder_hidden_states = encoder_hidden_states.to(hidden_states.device)
                if encoder_extended_attention_mask is not None:
                    encoder_extended_attention_mask = encoder_extended_attention_mask.to(hidden_states.device)
                if encoder_decoder_position_bias is not None:
                    encoder_decoder_position_bias = encoder_decoder_position_bias.to(hidden_states.device)
                if layer_head_mask is not None:
                    layer_head_mask = layer_head_mask.to(hidden_states.device)
                if cross_attn_layer_head_mask is not None:
                    cross_attn_layer_head_mask = cross_attn_layer_head_mask.to(hidden_states.device)

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                if use_cache:
                    logger.warning("`use_cache=True` is incompatible with gradient checkpointing. "
                                   "Setting `use_cache=False`...")
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return tuple(module(*inputs, use_cache, output_attentions))

                    return custom_forward

                layer_outputs = checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    extended_attention_mask,
                    input_node_ids,
                    input_edge_ids,
                    node_attention_mask,
                    edge_attention_mask,
                    extended_node_attention_mask,
                    extended_edge_attention_mask,
                    adj_matrix,
                    position_bias,
                    encoder_hidden_states,
                    encoder_extended_attention_mask,
                    encoder_decoder_position_bias,
                    layer_head_mask,
                    cross_attn_layer_head_mask,
                    None,  # past_key_value is always None with gradient checkpointing
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask=extended_attention_mask,
                    input_node_ids=input_node_ids,
                    input_edge_ids=input_edge_ids,
                    node_attention_mask=node_attention_mask,
                    edge_attention_mask=edge_attention_mask,
                    extended_node_attention_mask=extended_node_attention_mask,
                    extended_edge_attention_mask=extended_edge_attention_mask,
                    adj_matrix=adj_matrix,
                    position_bias=position_bias,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_extended_attention_mask,
                    encoder_decoder_position_bias=encoder_decoder_position_bias,
                    layer_head_mask=head_mask[i],
                    cross_attn_layer_head_mask=cross_attn_layer_head_mask,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )
            # layer_outputs is a tuple with:
            # hidden-states, key-value-states,
            # (self-attention weights), (self-attention position bias),
            # (cross-attention weights), (cross-attention position bias)
            if use_cache is False:
                layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]

            hidden_states, present_key_value_state = layer_outputs[:2]

            # if i == 0:
            # We share the position biases between the layers - the first layer store them
            # layer_outputs = hidden-states, key-value-states
            # (self-attention weights), (self-attention position bias),
            # (cross-attention weights), (cross-attention position bias)
            #  old version weight before bias, new version bias before weight,
            #         if output_attentions:     copied from attention code
            #             outputs = outputs + (weights,)
            #         if self.has_relative_attention_bias:
            #             outputs = outputs + (position_bias,),
            if i == 0:
                position_bias = layer_outputs[3 if output_attentions else 2]
                if self.is_decoder and encoder_hidden_states is not None:
                    encoder_decoder_position_bias = layer_outputs[5 if output_attentions else 3]
            # append next layer key value states
            present_key_value_states = present_key_value_states + (present_key_value_state,)

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[2],)  # We keep only self-attention weights for now
                if self.is_decoder:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[4],)
            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    present_key_value_states,
                    all_hidden_states,
                    all_attentions,
                    all_cross_attentions, ]
                if v is not None)
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=present_key_value_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )  # last-layer hidden state, (presents,) (all hidden states), (all attentions)


T5_START_DOCSTRING = r"""

    The T5 model was proposed in [Exploring the Limits of Transfer Learning with a Unified Text-to-Text
    Transformer](https://arxiv.org/abs/1910.10683) by Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan
    Narang, Michael Matena, Yanqi Zhou, Wei Li, Peter J. Liu. It's an encoder decoder transformer pre-trained in a
    text-to-text denoising generative setting.

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`T5Config`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

T5_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. T5 is a model with relative position embeddings so you
            should be able to pad the inputs on both the right and the left.

            Indices can be obtained using [`T5Tokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for detail.

            [What are input IDs?](../glossary#input-ids)

            To know more on how to prepare `input_ids` for pretraining take a look a [T5 Training](./t5#training).
        attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        decoder_input_ids (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Indices of decoder input sequence tokens in the vocabulary.

            Indices can be obtained using [`T5Tokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are decoder input IDs?](../glossary#decoder-input-ids)

            T5 uses the `pad_token_id` as the starting token for `decoder_input_ids` generation. If `past_key_values`
            is used, optionally only the last `decoder_input_ids` have to be input (see `past_key_values`).

            To know more on how to prepare `decoder_input_ids` for pretraining take a look at [T5
            Training](./t5#training).
        decoder_attention_mask (`torch.BoolTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also
            be used by default.
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules in the encoder. Mask values selected in `[0,
            1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        decoder_head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules in the decoder. Mask values selected in `[0,
            1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        cross_attn_head_mask (`torch.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
                Mask to nullify selected heads of the cross-attention modules in the decoder. Mask values selected in
                `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

        encoder_outputs (`tuple(tuple(torch.FloatTensor)`, *optional*):
            Tuple consists of (`last_hidden_state`, `optional`: *hidden_states*, `optional`: *attentions*)
            `last_hidden_state` of shape `(batch_size, sequence_length, hidden_size)` is a sequence of hidden states at
            the output of the last layer of the encoder. Used in the cross-attention of the decoder.
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        decoder_inputs_embeds (`torch.FloatTensor` of shape `(batch_size, target_sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `decoder_input_ids` you can choose to directly pass an embedded
            representation. If `past_key_values` is used, optionally only the last `decoder_inputs_embeds` have to be
            input (see `past_key_values`). This is useful if you want more control over how to convert
            `decoder_input_ids` indices into associated vectors than the model's internal embedding lookup matrix.

            If `decoder_input_ids` and `decoder_inputs_embeds` are both unset, `decoder_inputs_embeds` takes the value
            of `inputs_embeds`.

        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).

        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

T5_ENCODER_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. T5 is a model with relative position embeddings so you
            should be able to pad the inputs on both the right and the left.

            Indices can be obtained using [`T5Tokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for detail.

            To know more on how to prepare `input_ids` for pretraining take a look a [T5 Training](./t5#training).
        attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

# Warning message for FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
__HEAD_MASK_WARNING_MSG = """
The input argument `head_mask` was split into two arguments `head_mask` and `decoder_head_mask`. Currently,
`decoder_head_mask` is set to copy `head_mask`, but this feature is deprecated and will be removed in future versions.
If you do not want to use any `decoder_head_mask` now, please set `decoder_head_mask = torch.ones(num_layers,
num_heads)`.
"""


class T5Model(T5PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.shared)

        # Initialize weights and apply final processing
        self.post_init()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=None):
        self.device_map = (
            get_device_map(len(self.encoder.block), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.encoder.block))
        self.encoder.parallelize(self.device_map)
        self.decoder.parallelize(self.device_map)
        self.model_parallel = True

    @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
    def deparallelize(self):
        self.encoder.deparallelize()
        self.decoder.deparallelize()
        self.encoder = self.encoder.to("cpu")
        self.decoder = self.decoder.to("cpu")
        self.model_parallel = False
        self.device_map = None
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            encoder_outputs=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            decoder_past_key_valuess=None,
            use_cache=None,
            inputs_embeds=None,
            decoder_inputs_embeds=None,
            head_mask=None,
            output_attentions=None,
            output_hidden_states=None,
    ):
        r"""
    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.T5Config`) and inputs:
        last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
            If `decoder_past_key_valuess` is used only the last hidden-state of the sequences of shape :obj:`(batch_size, 1, hidden_size)` is output.
        decoder_past_key_valuess (:obj:`tuple(tuple(torch.FloatTensor))` of length :obj:`config.n_layers` with each tuple having 4 tensors of shape :obj:`(batch_size, num_heads, sequence_length, embed_size_per_head)`, `optional`, returned when ``use_cache=True``):
            Contains pre-computed key and value hidden-states of the attention blocks.
            Can be used to speed up sequential decoding (see `decoder_past_key_valuess` input).
            Note that when using `decoder_past_key_valuess`, the model only outputs the last `hidden-state` of the sequence of shape :obj:`(batch_size, 1, config.vocab_size)`.
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

        Example::

            >>> from transformers import T5Tokenizer, T5Model

            >>> tokenizer = T5Tokenizer.from_pretrained('t5-small')
            >>> model = T5Model.from_pretrained('t5-small')

            >>> input_ids = tokenizer.encode("Hello, my dog is cute", return_tensors="pt")  # Batch size 1
            >>> outputs = model(input_ids=input_ids, decoder_input_ids=input_ids)

            >>> last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple
        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

        hidden_states = encoder_outputs[0]

        # If decoding with past key value states, only the last tokens
        # should be given as an input
        if decoder_past_key_valuess is not None:
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids[:, -1:]
            if decoder_inputs_embeds is not None:
                decoder_inputs_embeds = decoder_inputs_embeds[:, -1:]

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_valuess=decoder_past_key_valuess,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        if use_cache is True:
            past = ((encoder_outputs, decoder_outputs[1]),)
            decoder_outputs = decoder_outputs[:1] + past + decoder_outputs[2:]

        return decoder_outputs + encoder_outputs


@add_start_docstrings("""T5 Model with a `language modeling` head on top.""", T5_START_DOCSTRING)
class T5ForConditionalGeneration(T5PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.model_dim = config.d_model

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        self.decoder = T5Stack(decoder_config, self.shared)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.init_weights()

    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=None):
        self.device_map = (
            get_device_map(len(self.encoder.block), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.encoder.block))
        self.encoder.parallelize(self.device_map)
        self.decoder.parallelize(self.device_map)
        self.lm_head = self.lm_head.to(self.decoder.first_device)
        self.model_parallel = True

    @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
    def deparallelize(self):
        self.encoder.deparallelize()
        self.decoder.deparallelize()
        self.encoder = self.encoder.to("cpu")
        self.decoder = self.decoder.to("cpu")
        self.lm_head = self.lm_head.to("cpu")
        self.model_parallel = False
        self.device_map = None
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_output_embeddings(self):
        return self.lm_head

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            encoder_outputs=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            decoder_past_key_valuess=None,
            use_cache=True,
            labels=None,
            inputs_embeds=None,
            decoder_inputs_embeds=None,
            head_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            **kwargs
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[-100, 0, ..., config.vocab_size - 1]`.
            All labels set to ``-100`` are ignored (masked), the loss is only
            computed for labels in ``[0, ..., config.vocab_size]``
        kwargs (:obj:`Dict[str, any]`, optional, defaults to `{}`):
            Used to hide legacy arguments that have been deprecated.

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.T5Config`) and inputs:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`labels` is provided):
            Classification loss (cross entropy).
        prediction_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`)
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
            If `past_key_valuess` is used only the last prediction_scores of the sequences of shape :obj:`(batch_size, 1, hidden_size)` is output.
        decoder_past_key_valuess (:obj:`tuple(tuple(torch.FloatTensor))` of length :obj:`config.n_layers` with each tuple having 4 tensors of shape :obj:`(batch_size, num_heads, sequence_length, embed_size_per_head)`, `optional`, returned when ``use_cache=True``):
            Contains pre-computed key and value hidden-states of the attention blocks.
            Can be used to speed up sequential decoding (see `decoder_past_key_valuess` input).
            Note that when using `decoder_past_key_valuess`, the model only outputs the last `prediction_score` of the sequence of shape :obj:`(batch_size, 1, config.vocab_size)`.
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

    Examples::

        >>> from transformers import T5Tokenizer, T5ForConditionalGeneration

        >>> tokenizer = T5Tokenizer.from_pretrained('t5-small')
        >>> model = T5ForConditionalGeneration.from_pretrained('t5-small')
        >>> input_ids = tokenizer.encode("Hello, my dog is cute", return_tensors="pt")  # Batch size 1
        >>> outputs = model(input_ids=input_ids, decoder_input_ids=input_ids, labels=input_ids)
        >>> loss, prediction_scores = outputs[:2]

        >>> tokenizer = T5Tokenizer.from_pretrained('t5-small')
        >>> model = T5ForConditionalGeneration.from_pretrained('t5-small')
        >>> input_ids = tokenizer.encode("summarize: Hello, my dog is cute", return_tensors="pt")  # Batch size 1
        >>> outputs = model.generate(input_ids)
        """

        if "lm_labels" in kwargs:
            warnings.warn(
                "The `lm_labels` argument is deprecated and will be removed in a future version, use `labels` instead.",
                DeprecationWarning,
            )
            labels = kwargs.pop("lm_labels")
        assert kwargs == {}, f"Unexpected keyword arguments: {list(kwargs.keys())}."

        use_cache = use_cache if use_cache is not None else self.config.use_cache

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

        hidden_states = encoder_outputs[0]

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # If decoding with past key value states, only the last tokens
        # should be given as an input
        if decoder_past_key_valuess is not None:
            assert labels is None, "Decoder should not use cached key value states when training."
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids[:, -1:]
            if decoder_inputs_embeds is not None:
                decoder_inputs_embeds = decoder_inputs_embeds[:, -1:]

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_valuess=decoder_past_key_valuess,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        # insert decoder past at right place
        # to speed up decoding
        if use_cache is True:
            past = ((encoder_outputs, decoder_outputs[1]),)
            decoder_outputs = decoder_outputs[:1] + past + decoder_outputs[2:]

        sequence_output = decoder_outputs[0]
        # Rescale output before projecting on vocab
        # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
        sequence_output = sequence_output * (self.model_dim ** -0.5)
        lm_logits = self.lm_head(sequence_output)

        decoder_outputs = (lm_logits,) + decoder_outputs[1:]  # Add hidden states and attention if they are here
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666
            decoder_outputs = (loss,) + decoder_outputs

        return decoder_outputs + encoder_outputs

    def prepare_inputs_for_generation(
            self,
            input_ids,
            past=None,
            attention_mask=None,
            head_mask=None,
            decoder_head_mask=None,
            cross_attn_head_mask=None,
            use_cache=None,
            encoder_outputs=None,
            **kwargs):
        # cut decoder_input_ids if past is used
        if past is not None:
            input_ids = input_ids[:, -1:]
        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache, }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return self._shift_right(labels)

    def _reorder_cache(self, past, beam_idx):
        # if decoder past is not included in output
        # speedy decoding is disabled and no need to reorder
        # if past[1] is None:
        #     logger.warning("You might want to consider setting `use_cache=True` to speed up decoding")
        #     return past
        # decoder_past = past[1]
        # past = (past[0],)
        # reordered_decoder_past = ()
        # for layer_past_states in decoder_past:
        #     # get the correct batch idx from layer past batch dim
        #     # batch dim of `past` is at 2nd position
        #     reordered_layer_past_states = ()
        #     for layer_past_state in layer_past_states:
        #         # need to set correct `past` for each of the four key / value states
        #         reordered_layer_past_states = reordered_layer_past_states + (
        #             layer_past_state.index_select(0, beam_idx),
        #         )
        #     assert reordered_layer_past_states[0].shape == layer_past_states[0].shape
        #     assert len(reordered_layer_past_states) == len(layer_past_states)
        #     reordered_decoder_past = reordered_decoder_past + (reordered_layer_past_states,)
        # return past + (reordered_decoder_past,)
        #  new version of beam code
        if past is None:
            logger.warning("You might want to consider setting `use_cache=True` to speed up decoding")
            return past
        reordered_decoder_past = ()
        for layer_past_states in past:
            # get the correct batch idx from layer past batch dim
            # batch dim of `past` is at 2nd position
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                # need to set correct `past` for each of the four key / value states
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(0, beam_idx.to(layer_past_state.device)),
                )
            assert reordered_layer_past_states[0].shape == layer_past_states[0].shape
            assert len(reordered_layer_past_states) == len(layer_past_states)
            reordered_decoder_past = reordered_decoder_past + (reordered_layer_past_states,)
        return reordered_decoder_past


class MyT5Pretrain(T5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.model_dim = config.d_model

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.use_cache = False
        self.encoder = T5StructStack(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        self.decoder = T5Stack(decoder_config, self.shared)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, encoder_outputs=None, encoder_label=None,
                decoder_input_ids=None, decoder_attention_mask=None, input_node_ids=None, input_edge_ids=None,
                node_length=None, edge_length=None, adj_matrix=None, decoder_whole_ids=None, word_length=None,
                decoder_past_key_valuess=None, use_cache=False, inputs_embeds=None, decoder_inputs_embeds=None,
                head_mask=None,
                output_attentions=None, output_hidden_states=None, is_training=False):

        if is_training:
            _decoder_input_ids = self._shift_right(decoder_input_ids)
        else:
            _decoder_input_ids = decoder_input_ids

        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                input_node_ids=input_node_ids,
                input_edge_ids=input_edge_ids,
                node_length=node_length,
                edge_length=edge_length,
                adj_matrix=adj_matrix,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
        hidden_states = encoder_outputs[0]

        # If decoding with past key value states, only the last tokens
        # should be given as an input
        if decoder_past_key_valuess is not None:
            # assert labels is None, "Decoder should not use cached key value states when training."
            assert is_training is False, "Decoder should not use cached key value states when training."
            if _decoder_input_ids is not None:
                _decoder_input_ids = _decoder_input_ids[:, -1:]
            if decoder_inputs_embeds is not None:
                decoder_inputs_embeds = decoder_inputs_embeds[:, -1:]

        # Decode
        decoder_outputs = self.decoder(
            input_ids=_decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_valuess=decoder_past_key_valuess,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        # insert decoder past at right place
        # to speed up decoding
        if use_cache is True:
            past = ((encoder_outputs, decoder_outputs[1]),)
            decoder_outputs = decoder_outputs[:1] + past + decoder_outputs[2:]

        sequence_output = decoder_outputs[0]
        # Rescale output before projecting on vocab
        # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
        sequence_output_rescale = sequence_output * (self.model_dim ** -0.5)
        hidden_states_rescale = hidden_states * (self.model_dim ** -0.5)
        lm_logits_decoder = self.lm_head(sequence_output_rescale)
        lm_logits_encoder = self.lm_head(hidden_states_rescale)

        decoder_outputs = (lm_logits_decoder,) + decoder_outputs[1:]  # Add hidden states and attention if they are here
        if encoder_label is None:
            if decoder_whole_ids is None:
                # Task 1: complete graph + corrupted text --> complete text
                if is_training:
                    loss_fct = nn.CrossEntropyLoss(ignore_index=self.config.pad_token_id)
                    loss = loss_fct(lm_logits_decoder.view(-1, self.config.vocab_size), decoder_input_ids.view(-1))
                    decoder_outputs = (loss,) + decoder_outputs
                    return loss
            else:
                # Task 3: complete graph + complete text --> ot loss
                if is_training:
                    # get each representation vector of nodes
                    node_attention_mask = invert_mask(convert_length_to_mask(node_length, input_node_ids.max() + 1))
                    edge_attention_mask = invert_mask(convert_length_to_mask(edge_length, input_edge_ids.max() + 1))
                    word_attention_mask = invert_mask(convert_length_to_mask(word_length, decoder_whole_ids.max() + 1))
                    outputs_node = scatter_mean(hidden_states, input_node_ids,
                                                dim=1)  # batch * node_size (50) * embed_dim
                    assert outputs_node.size(1) == node_attention_mask.size(1)

                    # get each representation vector of edges
                    outputs_edge = scatter_mean(hidden_states, input_edge_ids,
                                                dim=1)  # batch * edge_size (60) * embed_dim
                    assert outputs_edge.size(1) == edge_attention_mask.size(1)
                    graph_attention_mask = torch.cat((node_attention_mask, edge_attention_mask),
                                                     1)  # batch * (node_length+edge_length)
                    outputs_graph = torch.cat((outputs_node, outputs_edge),
                                              1)  # batch * (node_length+edge_length) * embed_dim

                    # get each representation vector of words
                    outputs_word = scatter_mean(sequence_output, decoder_whole_ids,
                                                dim=1)  # batch * output_length (128) * embed_dim
                    assert outputs_word.size(1) == word_attention_mask.size(1)
                    ot_dist = optimal_transport_dist(outputs_graph, outputs_word, graph_attention_mask,
                                                     word_attention_mask).to(outputs_graph)
                    loss = torch.mean(ot_dist)
                    return loss
        else:
            # Task 2: corrupted graph + complete text --> complete graph
            if is_training:
                loss_enc = nn.CrossEntropyLoss(ignore_index=-1)
                loss = loss_enc(lm_logits_encoder.view(-1, self.config.vocab_size), encoder_label.view(-1))
                return loss
        return decoder_outputs + encoder_outputs


class MyT5ForConditionalGeneration(T5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.model_dim = config.d_model

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5StructStack(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.shared)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.decode_tokenizer = None
        self.cls_tokenizer = None
        self.cls_model = None
        self.cls_start_step = None
        self.cls_threshold = .5
        # Initialize weights and apply final processing
        self.post_init()
        # Model parallel
        self.model_parallel = False
        self.device_map = None

    def forward(self,
                input_ids=None,
                attention_mask=None,
                decoder_input_ids=None,
                decoder_attention_mask=None,
                head_mask=None,
                decoder_head_mask=None,
                cross_attn_head_mask=None,
                encoder_outputs=None,
                past_key_values=None,
                input_node_ids=None, input_edge_ids=None,
                node_length=None, edge_length=None, adj_matrix=None,
                inputs_embeds=None,
                decoder_inputs_embeds=None,
                labels=None,
                use_cache=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                is_training=False):

        if is_training:
            _decoder_input_ids = self._shift_right(decoder_input_ids)
        else:
            _decoder_input_ids = decoder_input_ids

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                input_node_ids=input_node_ids,
                input_edge_ids=input_edge_ids,
                node_length=node_length,
                edge_length=edge_length,
                adj_matrix=adj_matrix,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict, )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)
        # If decoding with past key value states, only the last tokens
        # should be given as an input
        if past_key_values is not None:
            assert is_training is False, "Decoder should not use cached key value states when training."
            if _decoder_input_ids is not None:
                _decoder_input_ids = _decoder_input_ids[:, -1:]
            if decoder_inputs_embeds is not None:
                decoder_inputs_embeds = decoder_inputs_embeds[:, -1:]
        # Decode
        decoder_outputs = self.decoder(
            input_ids=_decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict, )
        # insert decoder past at right place
        # to speed up decoding
        #  deprecate cache here
        # if use_cache is True:
        #     past = ((encoder_outputs, decoder_outputs[1]),)
        #     decoder_outputs = decoder_outputs[:1] + past + decoder_outputs[2:]
        sequence_output = decoder_outputs[0]
        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)
        # Rescale output before projecting on vocab
        # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/
        # transformer/transformer.py#L586
        sequence_output_rescale = sequence_output * (self.model_dim ** -0.5)
        lm_logits_decoder = self.lm_head(sequence_output_rescale)
        # decoder_outputs = (lm_logits_decoder,) + decoder_outputs[1:]  # Add hidden states and attention
        # if they are here
        loss = None
        if decoder_input_ids is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.config.pad_token_id)
            loss = loss_fct(lm_logits_decoder.view(-1, self.config.vocab_size),
                            decoder_input_ids.view(-1))
        # if is_training:
        #     # decoder_outputs = (loss,) + decoder_outputs
        #     return loss

        if not return_dict:
            output = (lm_logits_decoder,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits_decoder,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions, )
#
#     @torch.no_grad()
#     def generate(self, inputs: Optional[torch.Tensor] = None, generation_config: Optional[GenerationConfig] = None,
#                  logits_processor: Optional[LogitsProcessorList] = None,
#                  stopping_criteria: Optional[StoppingCriteriaList] = None,
#                  prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
#                  synced_gpus: Optional[bool] = False,
#                  cls_prompt_list: Optional[list] = None,
#                  **kwargs, ) -> Union[GenerateOutput, torch.LongTensor]:
#         r"""
#         Generates sequences of token ids for models with a language modeling head.
#         <Tip warning={true}>
#         Most generation-controlling parameters are set in `generation_config` which, if not passed, will be set to the
#         model's default generation configuration. You can override any `generation_config` by passing the corresponding
#         parameters to generate(), e.g. `.generate(inputs, num_beams=4, do_sample=True)`.
#         For an overview of generation strategies and code examples, check out the [following
#         guide](./generation_strategies).
#         </Tip>
#         Parameters:
#             inputs (`torch.Tensor` of varying shape depending on the modality, *optional*):
#                 The sequence used as a prompt for the generation or as model inputs to the encoder. If `None` the
#                 method initializes it with `bos_token_id` and a batch size of 1. For decoder-only models `inputs`
#                 should of in the format of `input_ids`. For encoder-decoder models *inputs* can represent any of
#                 `input_ids`, `input_values`, `input_features`, or `pixel_values`.
#             generation_config (`~generation.GenerationConfig`, *optional*):
#                 The generation configuration to be used as base parametrization for the generation call. `**kwargs`
#                 passed to generate matching the attributes of `generation_config` will override them. If
#                 `generation_config` is not provided, the default will be used, which had the following loading
#                 priority: 1) from the `generation_config.json` model file, if it exists; 2) from the model
#                 configuration. Please note that unspecified parameters will inherit [`~generation.GenerationConfig`]'s
#                 default values, whose documentation should be checked to parameterize generation.
#             logits_processor (`LogitsProcessorList`, *optional*):
#                 Custom logits processors that complement the default logits processors built from arguments and
#                 generation config. If a logit processor is passed that is already created with the arguments or a
#                 generation config an error is thrown. This feature is intended for advanced users.
#             stopping_criteria (`StoppingCriteriaList`, *optional*):
#                 Custom stopping criteria that complement the default stopping criteria built from arguments and a
#                 generation config. If a stopping criteria is passed that is already created with the arguments or a
#                 generation config an error is thrown. This feature is intended for advanced users.
#             prefix_allowed_tokens_fn (`Callable[[int, torch.Tensor], List[int]]`, *optional*):
#                 If provided, this function constraints the beam search to allowed tokens only at each step. If not
#                 provided no constraint is applied. This function takes 2 arguments: the batch ID `batch_id` and
#                 `input_ids`. It has to return a list with the allowed tokens for the next generation step conditioned
#                 on the batch ID `batch_id` and the previously generated tokens `inputs_ids`. This argument is useful
#                 for constrained generation conditioned on the prefix, as described in [Autoregressive Entity
#                 Retrieval](https://arxiv.org/abs/2010.00904).
#             synced_gpus (`bool`, *optional*, defaults to `False`):
#                 Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
#             cls_prompt_list:
#                 List of constraints to be satisfied
#             kwargs:
#                 Ad hoc parametrization of `generate_config` and/or additional model-specific kwargs that will be
#                 forwarded to the `forward` function of the model. If the model is an encoder-decoder model, encoder
#                 specific kwargs should not be prefixed and decoder specific kwargs should be prefixed with *decoder_*.
#
#         Return:
#             [`~utils.ModelOutput`] or `torch.LongTensor`: A [`~utils.ModelOutput`] (if `return_dict_in_generate=True`
#             or when `config.return_dict_in_generate=True`) or a `torch.FloatTensor`.
#
#                 If the model is *not* an encoder-decoder model (`model.config.is_encoder_decoder=False`), the possible
#                 [`~utils.ModelOutput`] types are:
#
#                     - [`~generation.GreedySearchDecoderOnlyOutput`],
#                     - [`~generation.SampleDecoderOnlyOutput`],
#                     - [`~generation.BeamSearchDecoderOnlyOutput`],
#                     - [`~generation.BeamSampleDecoderOnlyOutput`]
#
#                 If the model is an encoder-decoder model (`model.config.is_encoder_decoder=True`), the possible
#                 [`~utils.ModelOutput`] types are:
#
#                     - [`~generation.GreedySearchEncoderDecoderOutput`],
#                     - [`~generation.SampleEncoderDecoderOutput`],
#                     - [`~generation.BeamSearchEncoderDecoderOutput`],
#                     - [`~generation.BeamSampleEncoderDecoderOutput`]
#         """
#         # 1. Handle `generation_config` and kwargs that might update it, and validate the `.generate()` call
#         self._validate_model_class()
#         # priority: `generation_config` argument > `model.generation_config` (the default generation config)
#         if generation_config is None:
#             # legacy: users may modify the model configuration to control generation -- update the generation config
#             # model attribute accordingly, if it was created from the model config
#             if self.generation_config._from_model_config:
#                 new_generation_config = GenerationConfig.from_model_config(self.config)
#                 if new_generation_config != self.generation_config:
#                     warnings.warn("You have modified the pretrained model configuration to control generation. This is "
#                                   "a deprecated strategy to control generation and will be removed soon, in a future "
#                                   "version. Please use a generation configuration file (see "
#                                   "https://huggingface.co/docs/transformers/main_classes/text_generation)")
#                     self.generation_config = new_generation_config
#             generation_config = self.generation_config
#         generation_config = copy.deepcopy(generation_config)
#         model_kwargs = generation_config.update(**kwargs)  # All unused kwargs must be model kwargs
#         self._validate_model_kwargs(model_kwargs.copy())
#         # 2. Set generation parameters if not already defined
#         logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
#         stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
#         if generation_config.pad_token_id is None and generation_config.eos_token_id is not None:
#             if model_kwargs.get("attention_mask", None) is None:
#                 logger.warning(
#                     "The attention mask and the pad token id were not set. As a consequence, you may observe "
#                     "unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.")
#             eos_token_id = generation_config.eos_token_id
#             if isinstance(eos_token_id, list):
#                 eos_token_id = eos_token_id[0]
#             logger.warning(f"Setting `pad_token_id` to `eos_token_id`:{eos_token_id} for open-end generation.")
#             generation_config.pad_token_id = eos_token_id
#         # 3. Define model inputs
#         # inputs_tensor has to be defined
#         # model_input_name is defined if model-specific keyword input is passed
#         # otherwise model_input_name is None
#         # all model-specific keyword inputs are removed from `model_kwargs`
#         inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(
#             inputs, generation_config.bos_token_id, model_kwargs)
#         batch_size = inputs_tensor.shape[0]
#         # 4. Define other model kwargs
#         model_kwargs["output_attentions"] = generation_config.output_attentions
#         model_kwargs["output_hidden_states"] = generation_config.output_hidden_states
#         model_kwargs["use_cache"] = generation_config.use_cache
#
#         accepts_attention_mask = "attention_mask" in set(inspect.signature(self.forward).parameters.keys())
#         requires_attention_mask = "encoder_outputs" not in model_kwargs
#         if model_kwargs.get("attention_mask", None) is None and requires_attention_mask and accepts_attention_mask:
#             model_kwargs["attention_mask"] = self._prepare_attention_mask_for_generation(
#                 inputs_tensor, generation_config.pad_token_id, generation_config.eos_token_id)
#         # decoder-only models should use left-padding for generation
#         if not self.config.is_encoder_decoder:
#             if (generation_config.pad_token_id is not None and
#                     torch.sum(inputs_tensor[:, -1] == generation_config.pad_token_id) > 0):
#                 logger.warning(
#                     "A decoder-only architecture is being used, but right-padding was detected! For correct "
#                     "generation results, please set `padding_side='left'` when initializing the tokenizer.")
#         if self.config.is_encoder_decoder and "encoder_outputs" not in model_kwargs:
#             # if model is encoder decoder encoder_outputs are created
#             # and added to `model_kwargs`
#             model_kwargs = self._prepare_encoder_decoder_kwargs_for_generation(inputs_tensor, model_kwargs,
#                                                                                model_input_name)
#         # 5. Prepare `input_ids` which will be used for auto-regressive generation
#         if self.config.is_encoder_decoder:
#             input_ids = self._prepare_decoder_input_ids_for_generation(
#                 batch_size, decoder_start_token_id=generation_config.decoder_start_token_id,
#                 bos_token_id=generation_config.bos_token_id, model_kwargs=model_kwargs, device=inputs_tensor.device, )
#         else:
#             # if decoder-only then inputs_tensor has to be `input_ids`
#             input_ids = inputs_tensor
#         # 6. Prepare `max_length` depending on other stopping criteria.
#         input_ids_seq_length = input_ids.shape[-1]
#         has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
#         if has_default_max_length and generation_config.max_new_tokens is None:
#             warnings.warn(f"Neither `max_length` nor `max_new_tokens` has been set, `max_length` will default to"
#                           f" {generation_config.max_length} (`generation_config.max_length`). Controlling `max_length` "
#                           "via the config is deprecated and `max_length` will be removed from the config in v5 of "
#                           "Transformers -- we recommend using `max_new_tokens` to control the maximum length of the "
#                           "generation.", UserWarning, )
#         elif has_default_max_length and generation_config.max_new_tokens is not None:
#             generation_config.max_length = generation_config.max_new_tokens + input_ids_seq_length
#         elif not has_default_max_length and generation_config.max_new_tokens is not None:
#             raise ValueError("Both `max_new_tokens` and `max_length` have been set but they serve the same purpose -- "
#                              "setting a limit to the generated output length. Remove one of those arguments. Please "
#                              "refer to the documentation for more information. "
#                              "(https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)")
#         if generation_config.min_length is not None and generation_config.min_length > generation_config.max_length:
#             raise ValueError(f"Unfeasible length constraints: the minimum length ({generation_config.min_length}) is "
#                              f"larger than the maximum length ({generation_config.max_length})")
#         if input_ids_seq_length >= generation_config.max_length:
#             input_ids_string = "decoder_input_ids" if self.config.is_encoder_decoder else "input_ids"
#             logger.warning(f"Input length of {input_ids_string} is {input_ids_seq_length}, but `max_length` is set to"
#                            f" {generation_config.max_length}. This can lead to unexpected behavior. You should consider"
#                            " increasing `max_new_tokens`.")
#         # 7. determine generation mode
#         is_constraint_gen_mode = (generation_config.constraints is not None or
#                                   generation_config.force_words_ids is not None)
#         is_contrastive_search_gen_mode = (generation_config.top_k is not None and generation_config.top_k > 1 and
#                                           generation_config.do_sample is False and
#                                           generation_config.penalty_alpha is not None and
#                                           generation_config.penalty_alpha > 0)
#         is_greedy_gen_mode = ((generation_config.num_beams == 1) and (generation_config.num_beam_groups == 1) and
#                               generation_config.do_sample is False and not is_constraint_gen_mode and
#                               not is_contrastive_search_gen_mode)
#         is_sample_gen_mode = ((generation_config.num_beams == 1) and (generation_config.num_beam_groups == 1) and
#                               generation_config.do_sample is True and not is_constraint_gen_mode and
#                               not is_contrastive_search_gen_mode)
#         is_beam_gen_mode = ((generation_config.num_beams > 1) and (generation_config.num_beam_groups == 1) and
#                             generation_config.do_sample is False and not is_constraint_gen_mode and
#                             not is_contrastive_search_gen_mode)
#         is_beam_sample_gen_mode = ((generation_config.num_beams > 1) and (generation_config.num_beam_groups == 1) and
#                                    generation_config.do_sample is True and not is_constraint_gen_mode and
#                                    not is_contrastive_search_gen_mode)
#         is_group_beam_gen_mode = ((generation_config.num_beams > 1) and (generation_config.num_beam_groups > 1) and
#                                   not is_constraint_gen_mode and not is_contrastive_search_gen_mode)
#         if generation_config.num_beam_groups > generation_config.num_beams:
#             raise ValueError("`num_beam_groups` has to be smaller or equal to `num_beams`")
#         if is_group_beam_gen_mode and generation_config.do_sample is True:
#             raise ValueError(
#                 "Diverse beam search cannot be used in sampling mode. Make sure that `do_sample` is set to `False`.")
#         if self.device.type != input_ids.device.type:
#             warnings.warn(
#                 "You are calling .generate() with the `input_ids` being on a device type different"
#                 f" than your model's device. `input_ids` is on {input_ids.device.type}, whereas the model"
#                 f" is on {self.device.type}. You may experience unexpected behaviors or slower generation."
#                 " Please make sure that you have put `input_ids` to the"
#                 f" correct device by calling for example input_ids = input_ids.to('{self.device.type}') before"
#                 " running `.generate()`.", UserWarning, )
#         # 8. prepare distribution pre_processing samplers
#         logits_processor = self._get_logits_processor(
#             generation_config=generation_config, input_ids_seq_length=input_ids_seq_length,
#             encoder_input_ids=inputs_tensor, prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
#             logits_processor=logits_processor, )
#         # 9. prepare stopping criteria
#         stopping_criteria = self._get_stopping_criteria(generation_config=generation_config,
#                                                         stopping_criteria=stopping_criteria)
#         # 10. go into different generation modes
#         if is_greedy_gen_mode:
#             if generation_config.num_return_sequences > 1:
#                 raise ValueError(f"num_return_sequences has to be 1, but is {generation_config.num_return_sequences} "
#                                  f"when doing greedy search.")
#             # 11. run greedy search
#             return self.greedy_search(input_ids, logits_processor=logits_processor, stopping_criteria=stopping_criteria,
#                                       pad_token_id=generation_config.pad_token_id,
#                                       eos_token_id=generation_config.eos_token_id,
#                                       output_scores=generation_config.output_scores,
#                                       return_dict_in_generate=generation_config.return_dict_in_generate,
#                                       synced_gpus=synced_gpus, **model_kwargs, )
#         elif is_contrastive_search_gen_mode:
#             if generation_config.num_return_sequences > 1:
#                 raise ValueError(f"num_return_sequences has to be 1, but is {generation_config.num_return_sequences} "
#                                  f"when doing contrastive search.")
#             return self.contrastive_search(input_ids, top_k=generation_config.top_k,
#                                            penalty_alpha=generation_config.penalty_alpha,
#                                            logits_processor=logits_processor, stopping_criteria=stopping_criteria,
#                                            pad_token_id=generation_config.pad_token_id,
#                                            eos_token_id=generation_config.eos_token_id,
#                                            output_scores=generation_config.output_scores,
#                                            return_dict_in_generate=generation_config.return_dict_in_generate,
#                                            synced_gpus=synced_gpus, **model_kwargs, )
#         elif is_sample_gen_mode:
#             # 11. prepare logits wrapper
#             logits_warper = self._get_logits_warper(generation_config)
#             # 12. expand input_ids with `num_return_sequences` additional sequences per batch
#             input_ids, model_kwargs = self._expand_inputs_for_generation(
#                 input_ids=input_ids, expand_size=generation_config.num_return_sequences,
#                 is_encoder_decoder=self.config.is_encoder_decoder, **model_kwargs, )
#             # 13. run sample
#             return self.sample(input_ids, logits_processor=logits_processor, logits_warper=logits_warper,
#                                stopping_criteria=stopping_criteria, pad_token_id=generation_config.pad_token_id,
#                                eos_token_id=generation_config.eos_token_id,
#                                output_scores=generation_config.output_scores,
#                                return_dict_in_generate=generation_config.return_dict_in_generate,
#                                synced_gpus=synced_gpus, **model_kwargs, )
#         elif is_beam_gen_mode:
#             if generation_config.num_return_sequences > generation_config.num_beams:
#                 raise ValueError("`num_return_sequences` has to be smaller or equal to `num_beams`.")
#             if stopping_criteria.max_length is None:
#                 raise ValueError("`max_length` needs to be a stopping_criteria for now.")
#             # 11. prepare beam search scorer
#             beam_scorer = ClassifierBeamScorer(batch_size=batch_size, num_beams=generation_config.num_beams,
#                                                device=inputs_tensor.device,
#                                                length_penalty=generation_config.length_penalty,
#                                                do_early_stopping=generation_config.early_stopping,
#                                                num_beam_hyps_to_keep=generation_config.num_return_sequences,
#                                                cls_prompt_list=cls_prompt_list,
#                                                decode_tokenizer=self.decode_tokenizer, cls_tokenizer=self.cls_tokenizer,
#                                                cls_model=self.cls_model, cls_threshold=self.cls_threshold,
#                                                cls_start_step=self.cls_start_step)
#             # 12. interleave input_ids with `num_beams` additional sequences per batch
#             input_ids, model_kwargs = self._expand_inputs_for_generation(
#                 input_ids=input_ids, expand_size=generation_config.num_beams,
#                 is_encoder_decoder=self.config.is_encoder_decoder, **model_kwargs, )
#             # 13. run beam search
#             return self.beam_search(
#                 input_ids, beam_scorer, logits_processor=logits_processor, stopping_criteria=stopping_criteria,
#                 pad_token_id=generation_config.pad_token_id, eos_token_id=generation_config.eos_token_id,
#                 output_scores=generation_config.output_scores,
#                 return_dict_in_generate=generation_config.return_dict_in_generate, synced_gpus=synced_gpus,
#                 **model_kwargs, )
#         elif is_beam_sample_gen_mode:
#             # 11. prepare logits warper
#             logits_warper = self._get_logits_warper(generation_config)
#             if stopping_criteria.max_length is None:
#                 raise ValueError("`max_length` needs to be a stopping_criteria for now.")
#             # 12. prepare beam search scorer
#             beam_scorer = ClassifierBeamScorer(batch_size=batch_size * generation_config.num_return_sequences,
#                                                num_beams=generation_config.num_beams, device=inputs_tensor.device,
#                                                length_penalty=generation_config.length_penalty,
#                                                do_early_stopping=generation_config.early_stopping,
#                                                cls_prompt_list=cls_prompt_list,
#                                                cls_threshold=self.cls_threshold, decode_tokenizer=self.decode_tokenizer,
#                                                cls_tokenizer=self.cls_tokenizer, cls_model=self.cls_model,
#                                                cls_start_step=self.cls_start_step)
#             beam_scorer.decode_tokenizer = self.decode_tokenizer
#             # 13. interleave input_ids with `num_beams` additional sequences per batch
#             input_ids, model_kwargs = self._expand_inputs_for_generation(
#                 input_ids=input_ids, expand_size=generation_config.num_beams * generation_config.num_return_sequences,
#                 is_encoder_decoder=self.config.is_encoder_decoder, **model_kwargs, )
#             # 14. run beam sample
#             return self.beam_sample(input_ids, beam_scorer, logits_processor=logits_processor,
#                                     logits_warper=logits_warper, stopping_criteria=stopping_criteria,
#                                     pad_token_id=generation_config.pad_token_id,
#                                     eos_token_id=generation_config.eos_token_id,
#                                     output_scores=generation_config.output_scores,
#                                     return_dict_in_generate=generation_config.return_dict_in_generate,
#                                     synced_gpus=synced_gpus, **model_kwargs, )
#         elif is_group_beam_gen_mode:
#             if generation_config.num_return_sequences > generation_config.num_beams:
#                 raise ValueError("`num_return_sequences` has to be smaller or equal to `num_beams`.")
#             if generation_config.num_beams % generation_config.num_beam_groups != 0:
#                 raise ValueError("`num_beams` should be divisible by `num_beam_groups` for group beam search.")
#             if stopping_criteria.max_length is None:
#                 raise ValueError("`max_length` needs to be a stopping_criteria for now.")
#             has_default_typical_p = kwargs.get("typical_p") is None and generation_config.typical_p == 1.0
#             if not has_default_typical_p:
#                 raise ValueError("Decoder argument `typical_p` is not supported with beam groups.")
#             # 11. prepare beam search scorer
#             beam_scorer = ClassifierBeamScorer(batch_size=batch_size, num_beams=generation_config.num_beams,
#                                                max_length=stopping_criteria.max_length, device=inputs_tensor.device,
#                                                length_penalty=generation_config.length_penalty,
#                                                do_early_stopping=generation_config.early_stopping,
#                                                num_beam_hyps_to_keep=generation_config.num_return_sequences,
#                                                num_beam_groups=generation_config.num_beam_groups,
#                                                cls_prompt_list=cls_prompt_list,
#                                                cls_threshold=self.cls_threshold, decode_tokenizer=self.decode_tokenizer,
#                                                cls_tokenizer=self.cls_tokenizer, cls_model=self.cls_model,
#                                                cls_start_step=self.cls_start_step)
#             # 12. interleave input_ids with `num_beams` additional sequences per batch
#             input_ids, model_kwargs = self._expand_inputs_for_generation(
#                 input_ids=input_ids, expand_size=generation_config.num_beams,
#                 is_encoder_decoder=self.config.is_encoder_decoder, **model_kwargs, )
#             # 13. run beam search
#             return self.group_beam_search(
#                 input_ids, beam_scorer, logits_processor=logits_processor, stopping_criteria=stopping_criteria,
#                 pad_token_id=generation_config.pad_token_id, eos_token_id=generation_config.eos_token_id,
#                 output_scores=generation_config.output_scores,
#                 return_dict_in_generate=generation_config.return_dict_in_generate, synced_gpus=synced_gpus,
#                 **model_kwargs, )
#         elif is_constraint_gen_mode:
#             if generation_config.num_return_sequences > generation_config.num_beams:
#                 raise ValueError("`num_return_sequences` has to be smaller or equal to `num_beams`.")
#             if stopping_criteria.max_length is None:
#                 raise ValueError("`max_length` needs to be a stopping_criteria for now.")
#             if generation_config.num_beams <= 1:
#                 raise ValueError("`num_beams` needs to be greater than 1 for constrained generation.")
#             if generation_config.do_sample:
#                 raise ValueError("`do_sample` needs to be false for constrained generation.")
#             if generation_config.num_beam_groups is not None and generation_config.num_beam_groups > 1:
#                 raise ValueError("`num_beam_groups` not supported yet for constrained generation.")
#             final_constraints = []
#             if generation_config.constraints is not None:
#                 final_constraints = generation_config.constraints
#             if generation_config.force_words_ids is not None:
#                 def typeerror():
#                     raise ValueError("`force_words_ids` has to either be a `List[List[List[int]]]` or `List[List[int]]`"
#                                      f"of positive integers, but is {generation_config.force_words_ids}.")
#
#                 if (not isinstance(generation_config.force_words_ids, list)
#                         or len(generation_config.force_words_ids) == 0):
#                     typeerror()
#                 for word_ids in generation_config.force_words_ids:
#                     if isinstance(word_ids[0], list):
#                         if not isinstance(word_ids, list) or len(word_ids) == 0:
#                             typeerror()
#                         if any(not isinstance(token_ids, list) for token_ids in word_ids):
#                             typeerror()
#                         if any(any((not isinstance(token_id, int) or token_id < 0) for token_id in token_ids)
#                                for token_ids in word_ids):
#                             typeerror()
#                         constraint = DisjunctiveConstraint(word_ids)
#                     else:
#                         if not isinstance(word_ids, list) or len(word_ids) == 0:
#                             typeerror()
#                         if any((not isinstance(token_id, int) or token_id < 0) for token_id in word_ids):
#                             typeerror()
#                         constraint = PhrasalConstraint(word_ids)
#                     final_constraints.append(constraint)
#             # 11. prepare beam search scorer
#             constrained_beam_scorer = ConstrainedBeamSearchScorer(
#                 constraints=final_constraints, batch_size=batch_size, num_beams=generation_config.num_beams,
#                 device=inputs_tensor.device, length_penalty=generation_config.length_penalty,
#                 do_early_stopping=generation_config.early_stopping,
#                 num_beam_hyps_to_keep=generation_config.num_return_sequences, )
#             # 12. interleave input_ids with `num_beams` additional sequences per batch
#             input_ids, model_kwargs = self._expand_inputs_for_generation(
#                 input_ids=input_ids, expand_size=generation_config.num_beams,
#                 is_encoder_decoder=self.config.is_encoder_decoder, **model_kwargs, )
#             # 13. run beam search
#             return self.constrained_beam_search(
#                 input_ids, constrained_beam_scorer=constrained_beam_scorer, logits_processor=logits_processor,
#                 stopping_criteria=stopping_criteria, pad_token_id=generation_config.pad_token_id,
#                 eos_token_id=generation_config.eos_token_id, output_scores=generation_config.output_scores,
#                 return_dict_in_generate=generation_config.return_dict_in_generate, synced_gpus=synced_gpus,
#                 **model_kwargs, )
#
#     def beam_search(self, input_ids: torch.LongTensor, beam_scorer: BeamScorer,
#                     logits_processor: Optional[LogitsProcessorList] = None,
#                     stopping_criteria: Optional[StoppingCriteriaList] = None,
#                     max_length: Optional[int] = None, pad_token_id: Optional[int] = None,
#                     eos_token_id: Optional[Union[int, List[int]]] = None,
#                     output_attentions: Optional[bool] = None,
#                     output_hidden_states: Optional[bool] = None, output_scores: Optional[bool] = None,
#                     return_dict_in_generate: Optional[bool] = None, synced_gpus: Optional[bool] = False,
#                     **model_kwargs, ) -> Union[BeamSearchOutput, torch.LongTensor]:
#         r"""Generates sequences of token ids for models with a language modeling head using **beam search decoding** and
#         can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.
#         <Tip warning={true}>
#         In most cases, you do not need to call [`~generation.GenerationMixin.beam_search`] directly. Use generate()
#         instead. For an overview of generation strategies and code examples, check the [following guide]
#         (./generation_strategies).
#         </Tip>
#
#         Parameters:
#             input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
#                 The sequence used as a prompt for the generation.
#             beam_scorer (`BeamScorer`):
#                 An derived instance of [`BeamScorer`] that defines how beam hypotheses are constructed, stored and
#                 sorted during generation. For more information, the documentation of [`BeamScorer`] should be read.
#             logits_processor (`LogitsProcessorList`, *optional*):
#                 An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
#                 used to modify the prediction scores of the language modeling head applied at each generation step.
#             stopping_criteria (`StoppingCriteriaList`, *optional*):
#                 An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
#                 used to tell if the generation loop should stop.
#             max_length (`int`, *optional*, defaults to 20):
#                 **DEPRECATED**. Use `logits_processor` or `stopping_criteria` directly to cap the number of generated
#                 tokens. The maximum length of the sequence to be generated.
#             pad_token_id (`int`, *optional*):
#                 The id of the *padding* token.
#             eos_token_id (`int`, *optional*):
#                 The id of the *end-of-sequence* token.
#             output_attentions (`bool`, *optional*, defaults to `False`):
#                 Whether or not to return the attentions tensors of all attention layers. See `attentions` under
#                 returned tensors for more details.
#             output_hidden_states (`bool`, *optional*, defaults to `False`):
#                 Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
#                 for more details.
#             output_scores (`bool`, *optional*, defaults to `False`):
#                 Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
#             return_dict_in_generate (`bool`, *optional*, defaults to `False`):
#                 Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
#             synced_gpus (`bool`, *optional*, defaults to `False`):
#                 Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
#             model_kwargs:
#                 Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
#                 an encoder-decoder model the kwargs should include `encoder_outputs`.
#         Return:
#             [`generation.BeamSearchDecoderOnlyOutput`], [`~generation.BeamSearchEncoderDecoderOutput`] or
#             `torch.LongTensor`: A `torch.LongTensor` containing the generated tokens (default behaviour) or a
#             [`~generation.BeamSearchDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
#             `return_dict_in_generate=True` or a [`~generation.BeamSearchEncoderDecoderOutput`] if
#             `model.config.is_encoder_decoder=True`.
#         Examples: ```python
#         >>> from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM, LogitsProcessorList,
#         ...     MinLengthLogitsProcessor, BeamSearchScorer, )
#         >>> import torch
#         >>> tokenizer = AutoTokenizer.from_pretrained("t5-base")
#         >>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
#         >>> encoder_input_str = "translate English to German: How old are you?"
#         >>> encoder_input_ids = tokenizer(encoder_input_str, return_tensors="pt").input_ids
#         >>> # lets run beam search using 3 beams
#         >>> num_beams = 3
#         >>> # define decoder start token ids
#         >>> input_ids = torch.ones((num_beams, 1), device=model.device, dtype=torch.long)
#         >>> input_ids = input_ids * model.config.decoder_start_token_id
#         >>> # add encoder_outputs to model keyword arguments
#         >>> model_kwargs = {"encoder_outputs": model.get_encoder()(
#         ...         encoder_input_ids.repeat_interleave(num_beams, dim=0), return_dict=True)}
#         >>> # instantiate beam scorer
#         >>> beam_scorer = BeamSearchScorer(batch_size=1, num_beams=num_beams, device=model.device, )
#         >>> # instantiate logits processors
#         >>> logits_processor = LogitsProcessorList(
#         ...     [MinLengthLogitsProcessor(5, eos_token_id=model.config.eos_token_id), ])
#         >>> outputs = model.beam_search(input_ids, beam_scorer, logits_processor=logits_processor, **model_kwargs)
#         >>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
#         ['Wie alt bist du?']```"""
#         # init values
#         logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
#         stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
#         if max_length is not None:
#             warnings.warn(
#                 "`max_length` is deprecated in this function, use "
#                 "`stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
#                 UserWarning,
#             )
#             stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
#         if len(stopping_criteria) == 0:
#             warnings.warn("You don't have defined any stopping_criteria, this will likely loop forever", UserWarning)
#         pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
#         eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
#         if isinstance(eos_token_id, int):
#             eos_token_id = [eos_token_id]
#         output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
#         output_attentions = (
#             output_attentions if output_attentions is not None else self.generation_config.output_attentions
#         )
#         output_hidden_states = (
#             output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
#         )
#         return_dict_in_generate = (
#             return_dict_in_generate
#             if return_dict_in_generate is not None
#             else self.generation_config.return_dict_in_generate
#         )
#
#         batch_size = len(beam_scorer._beam_hyps)
#         num_beams = beam_scorer.num_beams
#
#         batch_beam_size, cur_len = input_ids.shape
#
#         if num_beams * batch_size != batch_beam_size:
#             raise ValueError(
#                 f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."
#             )
#
#         # init attention / hidden states / scores tuples
#         scores = () if (return_dict_in_generate and output_scores) else None
#         beam_indices = (
#             tuple(() for _ in range(batch_beam_size)) if (return_dict_in_generate and output_scores) else None
#         )
#         decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
#         cross_attentions = () if (return_dict_in_generate and output_attentions) else None
#         decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None
#
#         # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
#         if return_dict_in_generate and self.config.is_encoder_decoder:
#             encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
#             encoder_hidden_states = (
#                 model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
#             )
#
#         # initialise score of first beam with 0 and the rest with -1e9. This makes sure that only tokens
#         # of the first beam are considered to avoid sampling the exact same tokens across all beams.
#         beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
#         beam_scores[:, 1:] = -1e9
#         beam_scores = beam_scores.view((batch_size * num_beams,))
#
#         this_peer_finished = False  # used by synced_gpus only
#         while True:
#             if synced_gpus:
#                 # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
#                 # The following logic allows an early break if all peers finished generating their sequence
#                 this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
#                 # send 0.0 if we finished, 1.0 otherwise
#                 dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
#                 # did all peers finish? the reduced sum will be 0.0 then
#                 if this_peer_finished_flag.item() == 0.0:
#                     break
#
#             model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
#
#             outputs = self(
#                 **model_inputs,
#                 return_dict=True,
#                 output_attentions=output_attentions,
#                 output_hidden_states=output_hidden_states,
#             )
#
#             if synced_gpus and this_peer_finished:
#                 cur_len = cur_len + 1
#                 continue  # don't waste resources running the code we don't need
#
#             next_token_logits = outputs.logits[:, -1, :]
#             # hack: adjust tokens for Marian. For Marian we have to make sure that the `pad_token_id`
#             # cannot be generated both before and after the `nn.functional.log_softmax` operation.
#             next_token_logits = self.adjust_logits_during_generation(next_token_logits, cur_len=cur_len)
#             # return log probability of softmax
#             next_token_scores = nn.functional.log_softmax(
#                 next_token_logits, dim=-1
#             )  # (batch_size * num_beams, vocab_size)
#             next_token_scores_processed = logits_processor(input_ids, next_token_scores)
#             # print(next_token_scores_processed)
#             next_token_scores = next_token_scores_processed + beam_scores[:, None].expand_as(next_token_scores)
#             # print(next_token_scores)
#             # Store scores, attentions and hidden_states when required
#             if return_dict_in_generate:
#                 if output_scores:
#                     scores += (next_token_scores_processed,)
#                 if output_attentions:
#                     decoder_attentions += (
#                         (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
#                     )
#                     if self.config.is_encoder_decoder:
#                         cross_attentions += (outputs.cross_attentions,)
#
#                 if output_hidden_states:
#                     decoder_hidden_states += (
#                         (outputs.decoder_hidden_states,)
#                         if self.config.is_encoder_decoder
#                         else (outputs.hidden_states,)
#                     )
#             # reshape for beam search
#             vocab_size = next_token_scores.shape[-1]
#             next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)
#             # Sample 2 next tokens for each beam (so we have some spare tokens and match output of beam search)
#             # how much should k be here?
#             next_token_scores, next_tokens = torch.topk(
#                 next_token_scores, 2 * num_beams, dim=1, largest=True, sorted=True
#             )
#
#             next_indices = torch.div(next_tokens, vocab_size, rounding_mode="floor")
#             next_tokens = next_tokens % vocab_size
#
#             # stateless
#             beam_outputs = beam_scorer.process(
#                 input_ids,
#                 next_token_scores,
#                 next_tokens,
#                 next_indices,
#                 pad_token_id=pad_token_id,
#                 eos_token_id=eos_token_id,
#                 beam_indices=beam_indices,
#                 current_step=cur_len)
#             beam_scores = beam_outputs["next_beam_scores"]
#             beam_next_tokens = beam_outputs["next_beam_tokens"]
#             beam_idx = beam_outputs["next_beam_indices"]
#             # print("beam result: top n_beam for each")
#             # print("logits:", beam_scores)
#             # print("tokens:", beam_next_tokens)
#             # print("from which beam", beam_idx)
#             # print([self.decode_tokenizer._convert_id_to_token(int(x)) for x in beam_next_tokens])
#             input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)
#             # for each_id_batch in input_ids:
#             #     print(self.decode_tokenizer.decode(each_id_batch))
#             # print()
#             model_kwargs = self._update_model_kwargs_for_generation(
#                 outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
#             )
#             if model_kwargs["past_key_values"] is not None:
#                 model_kwargs["past_key_values"] = self._reorder_cache(model_kwargs["past_key_values"], beam_idx)
#
#             if return_dict_in_generate and output_scores:
#                 beam_indices = tuple((beam_indices[beam_idx[i]] + (beam_idx[i],) for i in range(len(beam_indices))))
#
#             # increase cur_len
#             cur_len = cur_len + 1
#
#             if beam_scorer.is_done or stopping_criteria(input_ids, scores):
#                 if not synced_gpus:
#                     break
#                 else:
#                     this_peer_finished = True
#
#         sequence_outputs = beam_scorer.finalize(
#             input_ids,
#             beam_scores,
#             next_tokens,
#             next_indices,
#             pad_token_id=pad_token_id,
#             eos_token_id=eos_token_id,
#             max_length=stopping_criteria.max_length,
#             beam_indices=beam_indices,
#         )
#
#         if return_dict_in_generate:
#             if not output_scores:
#                 sequence_outputs["sequence_scores"] = None
#             if self.config.is_encoder_decoder:
#                 return BeamSearchEncoderDecoderOutput(
#                     sequences=sequence_outputs["sequences"],
#                     sequences_scores=sequence_outputs["sequence_scores"],
#                     scores=scores,
#                     beam_indices=sequence_outputs["beam_indices"],
#                     encoder_attentions=encoder_attentions,
#                     encoder_hidden_states=encoder_hidden_states,
#                     decoder_attentions=decoder_attentions,
#                     cross_attentions=cross_attentions,
#                     decoder_hidden_states=decoder_hidden_states,
#                 )
#             else:
#                 return BeamSearchDecoderOnlyOutput(
#                     sequences=sequence_outputs["sequences"],
#                     sequences_scores=sequence_outputs["sequence_scores"],
#                     scores=scores,
#                     beam_indices=sequence_outputs["beam_indices"],
#                     attentions=decoder_attentions,
#                     hidden_states=decoder_hidden_states,
#                 )
#         else:
#             return sequence_outputs["sequences"]
#
#     def beam_sample(
#             self,
#             input_ids: torch.LongTensor,
#             beam_scorer: BeamScorer,
#             logits_processor: Optional[LogitsProcessorList] = None,
#             stopping_criteria: Optional[StoppingCriteriaList] = None,
#             logits_warper: Optional[LogitsProcessorList] = None,
#             max_length: Optional[int] = None,
#             pad_token_id: Optional[int] = None,
#             eos_token_id: Optional[Union[int, List[int]]] = None,
#             output_attentions: Optional[bool] = None,
#             output_hidden_states: Optional[bool] = None,
#             output_scores: Optional[bool] = None,
#             return_dict_in_generate: Optional[bool] = None,
#             synced_gpus: Optional[bool] = False,
#             **model_kwargs,
#     ) -> Union[BeamSampleOutput, torch.LongTensor]:
#         r"""
#         Generates sequences of token ids for models with a language modeling head using **beam search multinomial
#         sampling** and can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.
#
#         <Tip warning={true}>
#
#         In most cases, you do not need to call [`~generation.GenerationMixin.beam_sample`] directly. Use generate()
#         instead. For an overview of generation strategies and code examples, check the [following
#         guide](./generation_strategies).
#
#         </Tip>
#
#         Parameters:
#             input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
#                 The sequence used as a prompt for the generation.
#             beam_scorer (`BeamScorer`):
#                 A derived instance of [`BeamScorer`] that defines how beam hypotheses are constructed, stored and
#                 sorted during generation. For more information, the documentation of [`BeamScorer`] should be read.
#             logits_processor (`LogitsProcessorList`, *optional*):
#                 An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
#                 used to modify the prediction scores of the language modeling head applied at each generation step.
#             stopping_criteria (`StoppingCriteriaList`, *optional*):
#                 An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
#                 used to tell if the generation loop should stop.
#             logits_warper (`LogitsProcessorList`, *optional*):
#                 An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsWarper`] used
#                 to warp the prediction score distribution of the language modeling head applied before multinomial
#                 sampling at each generation step.
#             max_length (`int`, *optional*, defaults to 20):
#                 **DEPRECATED**. Use `logits_processor` or `stopping_criteria` directly to cap the number of generated
#                 tokens. The maximum length of the sequence to be generated.
#             pad_token_id (`int`, *optional*):
#                 The id of the *padding* token.
#             eos_token_id (`int`, *optional*):
#                 The id of the *end-of-sequence* token.
#             output_attentions (`bool`, *optional*, defaults to `False`):
#                 Whether or not to return the attentions tensors of all attention layers. See `attentions` under
#                 returned tensors for more details.
#             output_hidden_states (`bool`, *optional*, defaults to `False`):
#                 Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
#                 for more details.
#             output_scores (`bool`, *optional*, defaults to `False`):
#                 Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
#             return_dict_in_generate (`bool`, *optional*, defaults to `False`):
#                 Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
#             synced_gpus (`bool`, *optional*, defaults to `False`):
#                 Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
#             model_kwargs:
#                 Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
#                 an encoder-decoder model the kwargs should include `encoder_outputs`.
#
#         Return:
#             [`~generation.BeamSampleDecoderOnlyOutput`], [`~generation.BeamSampleEncoderDecoderOutput`] or
#             `torch.LongTensor`: A `torch.LongTensor` containing the generated tokens (default behaviour) or a
#             [`~generation.BeamSampleDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
#             `return_dict_in_generate=True` or a [`~generation.BeamSampleEncoderDecoderOutput`] if
#             `model.config.is_encoder_decoder=True`.
#
#         Examples:
#
#         ```python
#         >>> from transformers import (
#         ...     AutoTokenizer,
#         ...     AutoModelForSeq2SeqLM,
#         ...     LogitsProcessorList,
#         ...     MinLengthLogitsProcessor,
#         ...     TopKLogitsWarper,
#         ...     TemperatureLogitsWarper,
#         ...     BeamSearchScorer,
#         ... )
#         >>> import torch
#
#         >>> tokenizer = AutoTokenizer.from_pretrained("t5-base")
#         >>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
#
#         >>> encoder_input_str = "translate English to German: How old are you?"
#         >>> encoder_input_ids = tokenizer(encoder_input_str, return_tensors="pt").input_ids
#
#         >>> # lets run beam search using 3 beams
#         >>> num_beams = 3
#         >>> # define decoder start token ids
#         >>> input_ids = torch.ones((num_beams, 1), device=model.device, dtype=torch.long)
#         >>> input_ids = input_ids * model.config.decoder_start_token_id
#
#         >>> # add encoder_outputs to model keyword arguments
#         >>> model_kwargs = {
#         ...     "encoder_outputs": model.get_encoder()(
#         ...         encoder_input_ids.repeat_interleave(num_beams, dim=0), return_dict=True
#         ...     )
#         ... }
#
#         >>> # instantiate beam scorer
#         >>> beam_scorer = BeamSearchScorer(
#         ...     batch_size=1,
#         ...     max_length=model.config.max_length,
#         ...     num_beams=num_beams,
#         ...     device=model.device,
#         ... )
#
#         >>> # instantiate logits processors
#         >>> logits_processor = LogitsProcessorList(
#         ...     [MinLengthLogitsProcessor(5, eos_token_id=model.config.eos_token_id)]
#         ... )
#         >>> # instantiate logits processors
#         >>> logits_warper = LogitsProcessorList(
#         ...     [
#         ...         TopKLogitsWarper(50),
#         ...         TemperatureLogitsWarper(0.7),
#         ...     ]
#         ... )
#
#         >>> outputs = model.beam_sample(
#         ...     input_ids, beam_scorer, logits_processor=logits_processor, logits_warper=logits_warper, **model_kwargs
#         ... )
#
#         >>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
#         ['Wie alt bist du?']
#         ```"""
#         # init values
#         logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
#         stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
#         if max_length is not None:
#             warnings.warn(
#                 "`max_length` is deprecated in this function, use"
#                 " `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
#                 UserWarning,
#             )
#             stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
#         pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
#         eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
#         if isinstance(eos_token_id, int):
#             eos_token_id = [eos_token_id]
#         output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
#         output_attentions = (
#             output_attentions if output_attentions is not None else self.generation_config.output_attentions
#         )
#         output_hidden_states = (
#             output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
#         )
#         return_dict_in_generate = (
#             return_dict_in_generate
#             if return_dict_in_generate is not None
#             else self.generation_config.return_dict_in_generate
#         )
#
#         batch_size = len(beam_scorer._beam_hyps)
#         num_beams = beam_scorer.num_beams
#
#         batch_beam_size, cur_len = input_ids.shape
#
#         # init attention / hidden states / scores tuples
#         scores = () if (return_dict_in_generate and output_scores) else None
#         beam_indices = (
#             tuple(() for _ in range(batch_beam_size)) if (return_dict_in_generate and output_scores) else None
#         )
#         decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
#         cross_attentions = () if (return_dict_in_generate and output_attentions) else None
#         decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None
#
#         # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
#         if return_dict_in_generate and self.config.is_encoder_decoder:
#             encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
#             encoder_hidden_states = (
#                 model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
#             )
#
#         beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
#         beam_scores = beam_scores.view((batch_size * num_beams,))
#         # used by synced_gpus only
#         this_peer_finished = False
#         while True:
#             if synced_gpus:
#                 # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
#                 # The following logic allows an early break if all peers finished generating their sequence
#                 this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
#                 # send 0.0 if we finished, 1.0 otherwise
#                 dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
#                 # did all peers finish? the reduced sum will be 0.0 then
#                 if this_peer_finished_flag.item() == 0.0:
#                     break
#             model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
#             # print("input size", input_ids.size())
#             outputs = self(**model_inputs, return_dict=True, output_attentions=output_attentions,
#                            output_hidden_states=output_hidden_states, )
#             # print("output size", outputs.logits.size())
#
#             if synced_gpus and this_peer_finished:
#                 cur_len = cur_len + 1
#                 continue  # don't waste resources running the code we don't need
#             next_token_logits = outputs.logits[:, -1, :]
#             # hack: adjust tokens for Marian. For Marian we have to make sure that the `pad_token_id`
#             # cannot be generated both before and after the `nn.functional.log_softmax` operation.
#             next_token_logits = self.adjust_logits_during_generation(next_token_logits, cur_len=cur_len)
#             # (batch_size * num_beams, vocab_size)
#             next_token_scores = nn.functional.log_softmax(next_token_logits, dim=-1)
#             next_token_scores_processed = logits_processor(input_ids, next_token_scores)
#             next_token_scores = next_token_scores_processed + beam_scores[:, None].expand_as(next_token_scores)
#             next_token_scores = logits_warper(input_ids, next_token_scores)
#             # Store scores, attentions and hidden_states when required
#             if return_dict_in_generate:
#                 if output_scores:
#                     scores += (logits_warper(input_ids, next_token_scores_processed),)
#                 if output_attentions:
#                     decoder_attentions += (
#                         (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,))
#                     if self.config.is_encoder_decoder:
#                         cross_attentions += (outputs.cross_attentions,)
#                 if output_hidden_states:
#                     decoder_hidden_states += ((outputs.decoder_hidden_states,) if self.config.is_encoder_decoder
#                                               else (outputs.hidden_states,))
#             # reshape for beam search
#             vocab_size = next_token_scores.shape[-1]
#             # print("size vocab", vocab_size)
#             next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)
#             probs = nn.functional.softmax(next_token_scores, dim=-1)
#             next_tokens = torch.multinomial(probs, num_samples=2 * num_beams)
#             # print("prob size", probs.size())
#             next_token_scores = torch.gather(next_token_scores, -1, next_tokens)
#             next_token_scores, _indices = torch.sort(next_token_scores, descending=True, dim=1)
#             next_tokens = torch.gather(next_tokens, -1, _indices)
#             next_indices = torch.div(next_tokens, vocab_size, rounding_mode="floor")
#             next_tokens = next_tokens % vocab_size
#
#             # stateless
#             beam_outputs = beam_scorer.process(
#                 input_ids,
#                 next_token_scores,
#                 next_tokens,
#                 next_indices,
#                 pad_token_id=pad_token_id,
#                 eos_token_id=eos_token_id,
#                 beam_indices=beam_indices,
#                 current_step=cur_len)
#
#             beam_scores = beam_outputs["next_beam_scores"]
#             beam_next_tokens = beam_outputs["next_beam_tokens"]
#             beam_idx = beam_outputs["next_beam_indices"]
#
#             # print("beam result: top n_beam for each")
#             # print("logits:", beam_scores)
#             # print("tokens:", beam_next_tokens)
#             # print("from which beam", beam_idx)
#             # print([self.decode_tokenizer._convert_id_to_token(int(x)) for x in beam_next_tokens])
#
#             input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)
#             # decoded: torch.LongTensor = input_ids.new(batch_size * 1,
#             #                                           stopping_criteria.max_length)
#             # for each_id_batch in input_ids:
#             #     print(self.decode_tokenizer.decode(each_id_batch))
#             # print()
#             model_kwargs = self._update_model_kwargs_for_generation(
#                 outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
#             )
#             if model_kwargs["past_key_values"] is not None:
#                 model_kwargs["past_key_values"] = self._reorder_cache(model_kwargs["past_key_values"], beam_idx)
#
#             if return_dict_in_generate and output_scores:
#                 beam_indices = tuple((beam_indices[beam_idx[i]] + (beam_idx[i],) for i in range(len(beam_indices))))
#
#             # increase cur_len
#             cur_len = cur_len + 1
#
#             if beam_scorer.is_done or stopping_criteria(input_ids, scores):
#                 if not synced_gpus:
#                     break
#                 else:
#                     this_peer_finished = True
#
#         sequence_outputs = beam_scorer.finalize(
#             input_ids,
#             beam_scores,
#             next_tokens,
#             next_indices,
#             pad_token_id=pad_token_id,
#             eos_token_id=eos_token_id,
#             max_length=stopping_criteria.max_length,
#             beam_indices=beam_indices,
#         )
#
#         if return_dict_in_generate:
#             if not output_scores:
#                 sequence_outputs["sequence_scores"] = None
#
#             if self.config.is_encoder_decoder:
#                 return BeamSampleEncoderDecoderOutput(
#                     sequences=sequence_outputs["sequences"],
#                     sequences_scores=sequence_outputs["sequence_scores"],
#                     scores=scores,
#                     beam_indices=sequence_outputs["beam_indices"],
#                     encoder_attentions=encoder_attentions,
#                     encoder_hidden_states=encoder_hidden_states,
#                     decoder_attentions=decoder_attentions,
#                     cross_attentions=cross_attentions,
#                     decoder_hidden_states=decoder_hidden_states,
#                 )
#             else:
#                 return BeamSampleDecoderOnlyOutput(
#                     sequences=sequence_outputs["sequences"],
#                     sequences_scores=sequence_outputs["sequence_scores"],
#                     scores=scores,
#                     beam_indices=sequence_outputs["beam_indices"],
#                     attentions=decoder_attentions,
#                     hidden_states=decoder_hidden_states,
#                 )
#         else:
#             return sequence_outputs["sequences"]
#
#     def group_beam_search(
#             self,
#             input_ids: torch.LongTensor,
#             beam_scorer: BeamScorer,
#             logits_processor: Optional[LogitsProcessorList] = None,
#             stopping_criteria: Optional[StoppingCriteriaList] = None,
#             max_length: Optional[int] = None,
#             pad_token_id: Optional[int] = None,
#             eos_token_id: Optional[Union[int, List[int]]] = None,
#             output_attentions: Optional[bool] = None,
#             output_hidden_states: Optional[bool] = None,
#             output_scores: Optional[bool] = None,
#             return_dict_in_generate: Optional[bool] = None,
#             synced_gpus: Optional[bool] = False,
#             **model_kwargs,
#     ):
#         r"""
#         Generates sequences of token ids for models with a language modeling head using **diverse beam search
#         decoding** and can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.
#
#         <Tip warning={true}>
#
#         In most cases, you do not need to call [`~generation.GenerationMixin.group_beam_search`] directly. Use
#         generate() instead. For an overview of generation strategies and code examples, check the [following
#         guide](./generation_strategies).
#
#         </Tip>
#
#         Parameters:
#             input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
#                 The sequence used as a prompt for the generation.
#             beam_scorer (`BeamScorer`):
#                 An derived instance of [`BeamScorer`] that defines how beam hypotheses are constructed, stored and
#                 sorted during generation. For more information, the documentation of [`BeamScorer`] should be read.
#             logits_processor (`LogitsProcessorList`, *optional*):
#                 An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
#                 used to modify the prediction scores of the language modeling head applied at each generation step.
#             stopping_criteria (`StoppingCriteriaList`, *optional*):
#                 An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
#                 used to tell if the generation loop should stop.
#             max_length (`int`, *optional*, defaults to 20):
#                 **DEPRECATED**. Use `logits_processor` or `stopping_criteria` directly to cap the number of generated
#                 tokens. The maximum length of the sequence to be generated.
#             pad_token_id (`int`, *optional*):
#                 The id of the *padding* token.
#             eos_token_id (`int`, *optional*):
#                 The id of the *end-of-sequence* token.
#             output_attentions (`bool`, *optional*, defaults to `False`):
#                 Whether or not to return the attentions tensors of all attention layers. See `attentions` under
#                 returned tensors for more details.
#             output_hidden_states (`bool`, *optional*, defaults to `False`):
#                 Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
#                 for more details.
#             output_scores (`bool`, *optional*, defaults to `False`):
#                 Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
#             return_dict_in_generate (`bool`, *optional*, defaults to `False`):
#                 Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
#             synced_gpus (`bool`, *optional*, defaults to `False`):
#                 Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
#
#             model_kwargs:
#                 Additional model specific kwargs that will be forwarded to the `forward` function of the model. If
#                 model is an encoder-decoder model the kwargs should include `encoder_outputs`.
#
#         Return:
#             [`~generation.BeamSearchDecoderOnlyOutput`], [`~generation.BeamSearchEncoderDecoderOutput`] or
#             `torch.LongTensor`: A `torch.LongTensor` containing the generated tokens (default behaviour) or a
#             [`~generation.BeamSearchDecoderOnlyOutput`] if [`~generation.BeamSearchDecoderOnlyOutput`] if
#             `model.config.is_encoder_decoder=False` and `return_dict_in_generate=True` or a
#             [`~generation.BeamSearchEncoderDecoderOutput`] if `model.config.is_encoder_decoder=True`.
#
#         Examples:
#
#         ```python
#         >>> from transformers import (
#         ...     AutoTokenizer,
#         ...     AutoModelForSeq2SeqLM,
#         ...     LogitsProcessorList,
#         ...     MinLengthLogitsProcessor,
#         ...     HammingDiversityLogitsProcessor,
#         ...     BeamSearchScorer,
#         ... )
#         >>> import torch
#
#         >>> tokenizer = AutoTokenizer.from_pretrained("t5-base")
#         >>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
#
#         >>> encoder_input_str = "translate English to German: How old are you?"
#         >>> encoder_input_ids = tokenizer(encoder_input_str, return_tensors="pt").input_ids
#
#
#         >>> # lets run diverse beam search using 6 beams
#         >>> num_beams = 6
#         >>> # define decoder start token ids
#         >>> input_ids = torch.ones((num_beams, 1), device=model.device, dtype=torch.long)
#         >>> input_ids = input_ids * model.config.decoder_start_token_id
#
#         >>> # add encoder_outputs to model keyword arguments
#         >>> model_kwargs = {
#         ...     "encoder_outputs": model.get_encoder()(
#         ...         encoder_input_ids.repeat_interleave(num_beams, dim=0), return_dict=True
#         ...     )
#         ... }
#
#         >>> # instantiate beam scorer
#         >>> beam_scorer = BeamSearchScorer(
#         ...     batch_size=1,
#         ...     max_length=model.config.max_length,
#         ...     num_beams=num_beams,
#         ...     device=model.device,
#         ...     num_beam_groups=3,
#         ... )
#
#         >>> # instantiate logits processors
#         >>> logits_processor = LogitsProcessorList(
#         ...     [
#         ...         HammingDiversityLogitsProcessor(5.5, num_beams=6, num_beam_groups=3),
#         ...         MinLengthLogitsProcessor(5, eos_token_id=model.config.eos_token_id),
#         ...     ]
#         ... )
#
#         >>> outputs = model.group_beam_search(
#         ...     input_ids, beam_scorer, logits_processor=logits_processor, **model_kwargs
#         ... )
#
#         >>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
#         ['Wie alt bist du?']
#         ```"""
#         # init values
#         logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
#         stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
#         if max_length is not None:
#             warnings.warn(
#                 "`max_length` is deprecated in this function, use"
#                 " `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
#                 UserWarning,
#             )
#             stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
#         pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
#         eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
#         if isinstance(eos_token_id, int):
#             eos_token_id = [eos_token_id]
#         output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
#         output_attentions = (
#             output_attentions if output_attentions is not None else self.generation_config.output_attentions
#         )
#         output_hidden_states = (
#             output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
#         )
#         return_dict_in_generate = (
#             return_dict_in_generate
#             if return_dict_in_generate is not None
#             else self.generation_config.return_dict_in_generate
#         )
#
#         batch_size = len(beam_scorer._beam_hyps)
#         num_beams = beam_scorer.num_beams
#         num_beam_groups = beam_scorer.num_beam_groups
#         num_sub_beams = num_beams // num_beam_groups
#         device = input_ids.device
#
#         batch_beam_size, cur_len = input_ids.shape
#
#         if return_dict_in_generate and output_scores:
#             beam_indices = [tuple(() for _ in range(num_sub_beams * batch_size)) for _ in range(num_beam_groups)]
#         else:
#             beam_indices = None
#
#         if num_beams * batch_size != batch_beam_size:
#             raise ValueError(
#                 f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."
#             )
#
#         # init attention / hidden states / scores tuples
#         scores = () if (return_dict_in_generate and output_scores) else None
#         decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
#         cross_attentions = () if (return_dict_in_generate and output_attentions) else None
#         decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None
#
#         # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
#         if return_dict_in_generate and self.config.is_encoder_decoder:
#             encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
#             encoder_hidden_states = (
#                 model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
#             )
#
#         # initialise score of first beam of each group with 0 and the rest with -1e9. This ensures that the beams in
#         # the same group don't produce same tokens everytime.
#         beam_scores = torch.full((batch_size, num_beams), -1e9, dtype=torch.float, device=device)
#         beam_scores[:, ::num_sub_beams] = 0
#         beam_scores = beam_scores.view((batch_size * num_beams,))
#
#         this_peer_finished = False  # used by synced_gpus only
#         while True:
#             if synced_gpus:
#                 # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
#                 # The following logic allows an early break if all peers finished generating their sequence
#                 this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
#                 # send 0.0 if we finished, 1.0 otherwise
#                 dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
#                 # did all peers finish? the reduced sum will be 0.0 then
#                 if this_peer_finished_flag.item() == 0.0:
#                     break
#             # predicted tokens in cur_len step
#             current_tokens = torch.zeros(batch_size * num_beams, dtype=input_ids.dtype, device=device)
#
#             # indices which will form the beams in the next time step
#             reordering_indices = torch.zeros(batch_size * num_beams, dtype=torch.long, device=device)
#
#             # do one decoder step on all beams of all sentences in batch
#             model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
#             outputs = self(
#                 **model_inputs,
#                 return_dict=True,
#                 output_attentions=output_attentions,
#                 output_hidden_states=output_hidden_states,
#             )
#
#             if synced_gpus and this_peer_finished:
#                 cur_len = cur_len + 1
#                 continue  # don't waste resources running the code we don't need
#
#             if output_scores:
#                 processed_score = torch.zeros_like(outputs.logits[:, -1, :])
#
#             for beam_group_idx in range(num_beam_groups):
#                 group_start_idx = beam_group_idx * num_sub_beams
#                 group_end_idx = min(group_start_idx + num_sub_beams, num_beams)
#                 group_size = group_end_idx - group_start_idx
#
#                 # indices of beams of current group among all sentences in batch
#                 batch_group_indices = []
#
#                 for batch_idx in range(batch_size):
#                     batch_group_indices.extend(
#                         [batch_idx * num_beams + idx for idx in range(group_start_idx, group_end_idx)]
#                     )
#                 group_input_ids = input_ids[batch_group_indices]
#
#                 # select outputs of beams of current group only
#                 next_token_logits = outputs.logits[batch_group_indices, -1, :]
#
#                 # hack: adjust tokens for Marian. For Marian we have to make sure that the `pad_token_id`
#                 # cannot be generated both before and after the `nn.functional.log_softmax` operation.
#                 next_token_logits = self.adjust_logits_during_generation(next_token_logits, cur_len=cur_len)
#                 next_token_scores = nn.functional.log_softmax(
#                     next_token_logits, dim=-1
#                 )  # (batch_size * group_size, vocab_size)
#                 vocab_size = next_token_scores.shape[-1]
#
#                 next_token_scores_processed = logits_processor(
#                     group_input_ids, next_token_scores, current_tokens=current_tokens, beam_group_idx=beam_group_idx
#                 )
#                 next_token_scores = next_token_scores_processed + beam_scores[batch_group_indices].unsqueeze(-1)
#                 next_token_scores = next_token_scores.expand_as(next_token_scores_processed)
#
#                 if output_scores:
#                     processed_score[batch_group_indices] = next_token_scores_processed
#
#                 # reshape for beam search
#                 next_token_scores = next_token_scores.view(batch_size, group_size * vocab_size)
#
#                 # Sample 2 next tokens for each beam (so we have some spare tokens and match output of beam search)
#                 next_token_scores, next_tokens = torch.topk(
#                     next_token_scores, 2 * group_size, dim=1, largest=True, sorted=True
#                 )
#                 next_indices = torch.div(next_tokens, vocab_size, rounding_mode="floor")
#                 next_tokens = next_tokens % vocab_size
#                 # stateless
#                 process_beam_indices = sum(beam_indices, ()) if beam_indices is not None else None
#                 beam_outputs = beam_scorer.process(group_input_ids, next_token_scores, next_tokens, next_indices,
#                                                    pad_token_id=pad_token_id, eos_token_id=eos_token_id,
#                                                    beam_indices=process_beam_indices, current_step=cur_len)
#                 beam_scores[batch_group_indices] = beam_outputs["next_beam_scores"]
#                 beam_next_tokens = beam_outputs["next_beam_tokens"]
#                 beam_idx = beam_outputs["next_beam_indices"]
#                 if return_dict_in_generate and output_scores:
#                     beam_indices[beam_group_idx] = tuple(
#                         beam_indices[beam_group_idx][beam_idx[i]] + (beam_idx[i],) for i in range(len(beam_indices[0])))
#                 input_ids[batch_group_indices] = group_input_ids[beam_idx]
#                 group_input_ids = torch.cat([group_input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)
#                 current_tokens[batch_group_indices] = group_input_ids[:, -1]
#                 # (beam_idx // group_size) -> batch_idx
#                 # (beam_idx % group_size) -> offset of idx inside the group
#                 reordering_indices[batch_group_indices] = (
#                         num_beams * torch.div(beam_idx, group_size, rounding_mode="floor") + group_start_idx +
#                         (beam_idx % group_size))
#
#             # Store scores, attentions and hidden_states when required
#             if return_dict_in_generate:
#                 if output_scores:
#                     scores += (processed_score,)
#                 if output_attentions:
#                     decoder_attentions += (
#                         (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
#                     )
#                     if self.config.is_encoder_decoder:
#                         cross_attentions += (outputs.cross_attentions,)
#
#                 if output_hidden_states:
#                     decoder_hidden_states += (
#                         (outputs.decoder_hidden_states,)
#                         if self.config.is_encoder_decoder
#                         else (outputs.hidden_states,)
#                     )
#
#             input_ids = torch.cat([input_ids, current_tokens.unsqueeze(-1)], dim=-1)
#
#             model_kwargs = self._update_model_kwargs_for_generation(
#                 outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
#             )
#             if model_kwargs["past_key_values"] is not None:
#                 model_kwargs["past_key_values"] = self._reorder_cache(
#                     model_kwargs["past_key_values"], reordering_indices
#                 )
#
#             # increase cur_len
#             cur_len = cur_len + 1
#
#             if beam_scorer.is_done or stopping_criteria(input_ids, scores):
#                 if not synced_gpus:
#                     break
#                 else:
#                     this_peer_finished = True
#
#         final_beam_indices = sum(beam_indices, ()) if beam_indices is not None else None
#         sequence_outputs = beam_scorer.finalize(
#             input_ids,
#             beam_scores,
#             next_tokens,
#             next_indices,
#             pad_token_id=pad_token_id,
#             eos_token_id=eos_token_id,
#             max_length=stopping_criteria.max_length,
#             beam_indices=final_beam_indices,
#         )
#
#         if return_dict_in_generate:
#             if not output_scores:
#                 sequence_outputs["sequence_scores"] = None
#
#             if self.config.is_encoder_decoder:
#                 return BeamSearchEncoderDecoderOutput(
#                     sequences=sequence_outputs["sequences"],
#                     sequences_scores=sequence_outputs["sequence_scores"],
#                     scores=scores,
#                     beam_indices=sequence_outputs["beam_indices"],
#                     encoder_attentions=encoder_attentions,
#                     encoder_hidden_states=encoder_hidden_states,
#                     decoder_attentions=decoder_attentions,
#                     cross_attentions=cross_attentions,
#                     decoder_hidden_states=decoder_hidden_states,
#                 )
#             else:
#                 return BeamSearchDecoderOnlyOutput(
#                     sequences=sequence_outputs["sequences"],
#                     sequences_scores=sequence_outputs["sequence_scores"],
#                     scores=scores,
#                     beam_indices=sequence_outputs["beam_indices"],
#                     attentions=decoder_attentions,
#                     hidden_states=decoder_hidden_states,
#                 )
#         else:
#             return sequence_outputs["sequences"]
#
#
# class ClassifierBeamScorer(BeamScorer):
#     r"""[`BeamScorer`] implementing standard beam search decoding.
#     Adapted in part from [Facebook's XLM beam search
#     code](https://github.com/facebookresearch/XLM/blob/9e6f6814d17be4fe5b15f2e6c43eb2b2d76daeb4/src/model/transformer.py#L529).
#
#     Reference for the diverse beam search algorithm and implementation [Ashwin Kalyan's DBS
#     implementation](https://github.com/ashwinkalyan/dbs/blob/master/dbs/beam_utils.lua)
#     Args:
#         batch_size (`int`):
#             Batch Size of `input_ids` for which standard beam search decoding is run in parallel.
#         max_length (`int`):
#             The maximum length of the sequence to be generated.
#         num_beams (`int`):
#             Number of beams for beam search.
#         device (`torch.device`):
#             Defines the device type (*e.g.*, `"cpu"` or `"cuda"`) on which this instance of `BeamSearchScorer` will be
#             allocated.
#         length_penalty (`float`, *optional*, defaults to 1.0):
#             Exponential penalty to the length that is used with beam-based generation. It is applied as an exponent to
#             the sequence length, which in turn is used to divide the score of the sequence. Since the score is the log
#             likelihood of the sequence (i.e. negative), `length_penalty` > 0.0 promotes longer sequences, while
#             `length_penalty` < 0.0 encourages shorter sequences.
#         do_early_stopping (`bool`, *optional*, defaults to `False`):
#             Whether to stop the beam search when at least `num_beams` sentences are finished per batch or not.
#         num_beam_hyps_to_keep (`int`, *optional*, defaults to 1):
#             The number of beam hypotheses that shall be returned upon calling
#             [`~transformer.BeamSearchScorer.finalize`].
#         num_beam_groups (`int`):
#             Number of groups to divide `num_beams` into in order to ensure diversity among different groups of beams.
#             See [this paper](https://arxiv.org/pdf/1610.02424.pdf) for more details."""
#
#     def __init__(self, batch_size: int, num_beams: int, device: torch.device, length_penalty: Optional[float] = 1.0,
#                  do_early_stopping: Optional[bool] = False, num_beam_hyps_to_keep: Optional[int] = 1,
#                  num_beam_groups: Optional[int] = 1, cls_prompt_list: Optional[list] = None,
#                  decode_tokenizer: Optional[PreTrainedTokenizerBase] = None,
#                  cls_tokenizer: Optional[PreTrainedTokenizerBase] = None, cls_threshold: Optional[float] = .5,
#                  cls_model: Optional[PreTrainedModel] = None, cls_start_step: Optional[int] = 2, **kwargs, ):
#         self.num_beams = num_beams
#         self.device = device
#         self.length_penalty = length_penalty
#         self.do_early_stopping = do_early_stopping
#         self.num_beam_hyps_to_keep = num_beam_hyps_to_keep
#         self.num_beam_groups = num_beam_groups
#         self.group_size = self.num_beams // self.num_beam_groups
#         self._is_init = False
#         self._beam_hyps = [BeamHypotheses(num_beams=self.num_beams, length_penalty=self.length_penalty,
#                                           early_stopping=self.do_early_stopping, ) for _ in range(batch_size)]
#         self.cls_prompt_list = cls_prompt_list
#         self.decode_tokenizer = decode_tokenizer
#         self.cls_tokenizer = cls_tokenizer
#         self.cls_model = cls_model
#         # every probability is log in this function
#         self.cls_threshold = math.log(cls_threshold)
#         self.cls_start_step = cls_start_step
#         self._done = torch.tensor([False for _ in range(batch_size)], dtype=torch.bool, device=self.device)
#         if not isinstance(num_beams, int) or num_beams <= 1:
#             raise ValueError(f"`num_beams` has to be an integer strictly greater than 1, but is {num_beams}. For "
#                              f"`num_beams` == 1, one should make use of `greedy_search` instead.")
#         if not isinstance(num_beam_groups, int) or (num_beam_groups > num_beams) or (num_beams % num_beam_groups != 0):
#             raise ValueError(
#                 "`num_beam_groups` has to be an integer smaller or equal than `num_beams` and `num_beams` has to be"
#                 f" divisible by `num_beam_groups`, but is {num_beam_groups} with `num_beams` being {num_beams}.")
#         if "max_length" in kwargs:
#             warnings.warn("Passing `max_length` to BeamSearchScorer is deprecated and has no effect. `max_length` "
#                           "should be passed directly to `beam_search(...)`, `beam_sample(...)`, or "
#                           "`group_beam_search(...)`.")
#
#     @property
#     def is_done(self) -> bool:
#         return self._done.all()
#
#     def process(self, input_ids: torch.LongTensor, next_scores: torch.FloatTensor, next_tokens: torch.LongTensor,
#                 next_indices: torch.LongTensor, pad_token_id: Optional[int] = None,
#                 eos_token_id: Optional[Union[int, List[int]]] = None, beam_indices: Optional[torch.LongTensor] = None,
#                 current_step: Optional[int] = 0) -> Tuple[torch.Tensor]:
#         cur_len = input_ids.shape[-1]
#         batch_size = len(self._beam_hyps)
#         if not (batch_size == (input_ids.shape[0] // self.group_size)):
#             if self.num_beam_groups > 1:
#                 raise ValueError(f"A group beam size of {input_ids.shape[0]} is used as the input, but a group beam "
#                                  f"size of {self.group_size} is expected by the beam scorer.")
#             else:
#                 raise ValueError(f"A beam size of {input_ids.shape[0]} is used as the input, but a beam size of "
#                                  f"{self.group_size} is expected by the beam scorer.")
#         device = input_ids.device
#         next_beam_scores = torch.zeros((batch_size, self.group_size), dtype=next_scores.dtype, device=device)
#         next_beam_tokens = torch.zeros((batch_size, self.group_size), dtype=next_tokens.dtype, device=device)
#         next_beam_indices = torch.zeros((batch_size, self.group_size), dtype=next_indices.dtype, device=device)
#         if isinstance(eos_token_id, int):
#             eos_token_id = [eos_token_id]
#         for batch_idx, beam_hyp in enumerate(self._beam_hyps):
#             # deal with each batch
#             if self._done[batch_idx]:
#                 if self.num_beams < len(beam_hyp):
#                     raise ValueError(f"Batch can only be done if at least {self.num_beams} beams have been generated")
#                 if eos_token_id is None or pad_token_id is None:
#                     raise ValueError("Generated beams >= num_beams -> eos_token_id and pad_token have to be defined")
#                 # pad the batch
#                 next_beam_scores[batch_idx, :] = 0
#                 next_beam_tokens[batch_idx, :] = pad_token_id
#                 next_beam_indices[batch_idx, :] = 0
#                 continue
#             # print("token pool of 2*num_beam:", next_tokens[batch_idx])
#             # print("logit for each", next_scores)
#             # print("next indices", next_indices)
#             # print("beam indices", beam_indices)
#             # print("corresponding words",
#             #       [self.decode_tokenizer._convert_id_to_token(int(x)) for x in next_tokens[batch_idx]])
#             # print([self.decode_tokenizer.decode(x).replace("<pad>", "").replace("</s>", "").strip(".? ") for x in
#             #        torch.cat([input_ids[next_indices[batch_idx]], next_tokens[batch_idx].unsqueeze(-1)], 1)])
#             if current_step > self.cls_start_step:
#                 next_sentences = [self.decode_tokenizer.decode(x).replace("<pad>", "").replace("</s>", "")
#                                   for x in torch.cat([input_ids[next_indices[batch_idx]],
#                                                       next_tokens[batch_idx].unsqueeze(-1)], 1)]
#                 prev_sentences = [self.decode_tokenizer.decode(x).replace("<pad>", "").replace("</s>", "")
#                                   for x in input_ids[next_indices[batch_idx]]]
#                 classification_scores = torch.torch.zeros_like(next_scores[batch_idx], device=device)
#                 satisfied_sentence = torch.ones_like(next_scores[batch_idx], dtype=torch.bool, device=device)
#                 for each_prompt in self.cls_prompt_list:
#                     # make batch for classification
#                     cur_cls_inputs = [each_prompt[0] + each_candidate.strip(".? ")
#                                       for each_candidate in next_sentences]
#                     cur_tokenized_input = self.cls_tokenizer(cur_cls_inputs, truncation=True)
#                     cur_tokenized_input["input_ids"] = pad_sequence(
#                         [torch.LongTensor(x) for x in cur_tokenized_input["input_ids"]],
#                         padding_value=self.cls_tokenizer.pad_token_id, batch_first=True).to(device)
#                     cur_tokenized_input["attention_mask"] = pad_sequence(
#                         [torch.LongTensor(x) for x in cur_tokenized_input["attention_mask"]], padding_value=0,
#                         batch_first=True).to(device)
#                     cur_cls_output = self.cls_model(input_ids=cur_tokenized_input["input_ids"],
#                                                     attention_mask=cur_tokenized_input["attention_mask"])
#
#                     prev_cls_inputs = [each_prompt[0] + each_candidate.strip(".? ")
#                                        for each_candidate in prev_sentences]
#                     prev_tokenized_input = self.cls_tokenizer(prev_cls_inputs, truncation=True)
#                     prev_tokenized_input["input_ids"] = pad_sequence(
#                         [torch.LongTensor(x) for x in prev_tokenized_input["input_ids"]],
#                         padding_value=self.cls_tokenizer.pad_token_id, batch_first=True).to(device)
#                     prev_tokenized_input["attention_mask"] = pad_sequence(
#                         [torch.LongTensor(x) for x in prev_tokenized_input["attention_mask"]], padding_value=0,
#                         batch_first=True).to(device)
#                     prev_cls_output = self.cls_model(input_ids=prev_tokenized_input["input_ids"],
#                                                      attention_mask=prev_tokenized_input["attention_mask"])
#
#                     cur_log_cls = nn.functional.log_softmax(cur_cls_output.logits, dim=-1)[:, 1]
#                     satisfied_sentence = satisfied_sentence & torch.gt(cur_log_cls, self.cls_threshold)
#                     prev_cls = nn.functional.softmax(prev_cls_output.logits, dim=-1)[:, 1]
#                     # multiply different classifier results together
#                     # print(self.cls_threshold)
#                     # print(cur_log_cls)
#                     # print(cur_log_cls)
#                     # print(prev_cls)
#                     cur_log_cls = ((1 - prev_cls) ** 1) * cur_log_cls
#                     # cur_log_cls = torch.where(cur_log_cls > self.cls_threshold, 0, cur_log_cls)
#                     # print(cur_log_cls)
#                     # print(satisfied_sentence)
#                     classification_scores += cur_log_cls
#                     # print(classification_scores)
#                     # for idx in range(0, len(cur_cls_inputs)):
#                     #     print(cur_cls_inputs[idx])
#                     # print(cur_tokenized_input["input_ids"][idx])
#                     # print(cur_tokenized_input["attention_mask"][idx])
#                 # cls_scores = next_scores[batch_idx] + 2 * classification_scores
#                 # print(cls_scores)
#                 # re-sorting scores by classification
#                 # should we choose by best cls only, or combine them?
#                 candidate_scores = next_scores[batch_idx] + classification_scores / 20
#                 sorted_beam_cls, sorted_cls_index = torch.topk(candidate_scores, 2 * self.num_beams, dim=-1,
#                                                                largest=True, sorted=True)
#                 candidate_scores_sorted = torch.index_select(candidate_scores, dim=-1, index=sorted_cls_index)
#                 next_scores_sorted = torch.index_select(next_scores[batch_idx], dim=-1, index=sorted_cls_index)
#                 next_indices_sorted = torch.index_select(next_indices[batch_idx], dim=-1, index=sorted_cls_index)
#                 next_tokens_sorted = torch.index_select(next_tokens[batch_idx], dim=-1, index=sorted_cls_index)
#                 # print(sorted_beam_cls)
#                 # print(sorted_cls_index)
#                 # print("scores!", next_scores_sorted)
#                 # print(next_scores[batch_idx])
#                 # print("indices!", next_indices_sorted)
#                 # print(next_indices[batch_idx])
#                 # print("tokens!", next_tokens_sorted)
#                 # print(next_tokens[batch_idx])
#             else:
#                 satisfied_sentence = torch.zeros_like(next_scores[batch_idx], dtype=torch.bool, device=device)
#                 next_scores_sorted = candidate_scores_sorted = next_scores[batch_idx]
#                 next_indices_sorted = next_indices[batch_idx]
#                 next_tokens_sorted = next_tokens[batch_idx]
#             beam_idx = 0
#             for beam_token_rank, (next_token, next_score, next_index, boolean_satisfied, candidate_score) in enumerate(
#                     zip(next_tokens_sorted, next_scores_sorted, next_indices_sorted,
#                         satisfied_sentence, candidate_scores_sorted)):
#                 # zip(next_tokens[batch_idx], next_scores[batch_idx], next_indices[batch_idx])):
#                 # universal idx taking batch and beam group into consideration
#                 batch_beam_idx = batch_idx * self.group_size + next_index
#                 # add to generated hypotheses if end of sentenfce
#                 if (eos_token_id is not None) and (next_token.item() in eos_token_id):
#                 # if (eos_token_id is not None) and (next_token.item() in eos_token_id) and boolean_satisfied:
#                     # if beam_token does not belong to top num_beams tokens, it should not be added
#                     is_beam_token_worse_than_top_num_beams = beam_token_rank >= self.group_size
#                     if is_beam_token_worse_than_top_num_beams:
#                         continue
#                     if beam_indices is not None:
#                         beam_index = beam_indices[batch_beam_idx]
#                         beam_index = beam_index + (batch_beam_idx,)
#                     else:
#                         beam_index = None
#                     beam_hyp.add(input_ids[batch_beam_idx].clone(), candidate_score.item(), beam_indices=beam_index, )
#                     # beam_hyp.add(input_ids[batch_beam_idx].clone(), next_score.item(), beam_indices=beam_index, )
#                 else:
#                     # add next predicted token since it is not eos_token
#                     next_beam_scores[batch_idx, beam_idx] = next_score
#                     next_beam_tokens[batch_idx, beam_idx] = next_token
#                     next_beam_indices[batch_idx, beam_idx] = batch_beam_idx
#                     beam_idx += 1
#                 # once the beam for next step is full, don't add more tokens to it.
#                 if beam_idx == self.group_size:
#                     break
#             if beam_idx < self.group_size:
#                 raise ValueError(
#                     f"At most {self.group_size} tokens in {next_tokens_sorted} can be equal to `eos_token_id:"
#                     f" {eos_token_id}`. Make sure {next_tokens_sorted} are corrected.")
#
#             # Check if we are done so that we can save a pad step if all(done)
#             self._done[batch_idx] = self._done[batch_idx] or beam_hyp.is_done(
#                 next_scores[batch_idx].max().item(), cur_len
#             )
#         return UserDict({"next_beam_scores": next_beam_scores.view(-1), "next_beam_tokens": next_beam_tokens.view(-1),
#                          "next_beam_indices": next_beam_indices.view(-1), })
#
#     def finalize(
#             self,
#             input_ids: torch.LongTensor,
#             final_beam_scores: torch.FloatTensor,
#             final_beam_tokens: torch.LongTensor,
#             final_beam_indices: torch.LongTensor,
#             max_length: int,
#             pad_token_id: Optional[int] = None,
#             eos_token_id: Optional[Union[int, List[int]]] = None,
#             beam_indices: Optional[torch.LongTensor] = None,
#     ) -> Tuple[torch.LongTensor]:
#         batch_size = len(self._beam_hyps)
#         if isinstance(eos_token_id, int):
#             eos_token_id = [eos_token_id]
#         # finalize all open beam hypotheses and add to generated hypotheses
#         for batch_idx, beam_hyp in enumerate(self._beam_hyps):
#             if self._done[batch_idx]:
#                 continue
#             # all open beam hypotheses are added to the beam hypothesis
#             # beam hypothesis class automatically keeps the best beams
#             for beam_id in range(self.num_beams):
#                 batch_beam_idx = batch_idx * self.num_beams + beam_id
#                 final_score = final_beam_scores[batch_beam_idx].item()
#                 final_tokens = input_ids[batch_beam_idx]
#                 beam_index = beam_indices[batch_beam_idx] if beam_indices is not None else None
#                 beam_hyp.add(final_tokens, final_score, beam_indices=beam_index)
#         # select the best hypotheses
#         sent_lengths = input_ids.new(batch_size * self.num_beam_hyps_to_keep)
#         best = []
#         best_indices = []
#         best_scores = torch.zeros(batch_size * self.num_beam_hyps_to_keep, device=self.device, dtype=torch.float32)
#         # retrieve best hypotheses
#         for i, beam_hyp in enumerate(self._beam_hyps):
#             sorted_hyps = sorted(beam_hyp.beams, key=lambda x: x[0])
#             for j in range(self.num_beam_hyps_to_keep):
#                 best_hyp_tuple = sorted_hyps.pop()
#                 best_score = best_hyp_tuple[0]
#                 best_hyp = best_hyp_tuple[1]
#                 best_index = best_hyp_tuple[2]
#                 sent_lengths[self.num_beam_hyps_to_keep * i + j] = len(best_hyp)
#                 # append hyp to lists
#                 best.append(best_hyp)
#                 # append indices to list
#                 best_indices.append(best_index)
#                 best_scores[i * self.num_beam_hyps_to_keep + j] = best_score
#         # prepare for adding eos
#         sent_lengths_max = sent_lengths.max().item() + 1
#         sent_max_len = min(sent_lengths_max, max_length) if max_length is not None else sent_lengths_max
#         decoded: torch.LongTensor = input_ids.new(batch_size * self.num_beam_hyps_to_keep, sent_max_len)
#         if len(best_indices) > 0 and best_indices[0] is not None:
#             indices: torch.LongTensor = input_ids.new(batch_size * self.num_beam_hyps_to_keep, sent_max_len)
#         else:
#             indices = None
#         # shorter batches are padded if needed
#         if sent_lengths.min().item() != sent_lengths.max().item():
#             assert pad_token_id is not None, "`pad_token_id` has to be defined"
#             decoded.fill_(pad_token_id)
#         if indices is not None:
#             indices.fill_(-1)
#         # fill with hypotheses and eos_token_id if the latter fits in
#         for i, (hypo, best_idx) in enumerate(zip(best, best_indices)):
#             decoded[i, : sent_lengths[i]] = hypo
#             if indices is not None:
#                 indices[i, : len(best_idx)] = torch.tensor(best_idx)
#             if sent_lengths[i] < sent_max_len:
#                 # inserting only the first eos_token_id
#                 decoded[i, sent_lengths[i]] = eos_token_id[0]
#         return UserDict({"sequences": decoded, "sequence_scores": best_scores, "beam_indices": indices, })
#
#
# class BeamHypotheses:
#     def __init__(self, num_beams: int, length_penalty: float, early_stopping: bool):
#         """Initialize n-best list of hypotheses."""
#         self.length_penalty = length_penalty
#         self.early_stopping = early_stopping
#         self.num_beams = num_beams
#         self.beams = []
#         self.worst_score = 1e9
#
#     def __len__(self):
#         """Number of hypotheses in the list."""
#         return len(self.beams)
#
#     def add(self, hyp: torch.LongTensor, sum_logprobs: float, beam_indices: Optional[torch.LongTensor] = None):
#         """Add a new hypothesis to the list."""
#         score = sum_logprobs / (hyp.shape[-1] ** self.length_penalty)
#         if len(self) < self.num_beams or score > self.worst_score:
#             self.beams.append((score, hyp, beam_indices))
#             if len(self) > self.num_beams:
#                 sorted_next_scores = sorted([(s, idx) for idx, (s, _, _) in enumerate(self.beams)])
#                 del self.beams[sorted_next_scores[0][1]]
#                 self.worst_score = sorted_next_scores[1][0]
#             else:
#                 self.worst_score = min(score, self.worst_score)
#
#     def is_done(self, best_sum_logprobs: float, cur_len: int) -> bool:
#         """
#         If there are enough hypotheses and that none of the hypotheses being generated can become better than the worst
#         one in the heap, then we are done with this sentence.
#         """
#
#         if len(self) < self.num_beams:
#             return False
#         elif self.early_stopping:
#             return True
#         else:
#             cur_score = best_sum_logprobs / cur_len ** self.length_f
#             ret = self.worst_score >= cur_score
#             return ret
