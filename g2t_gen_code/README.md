
## G2T Generation Codes

This folder is a PyTorch implementation of several graph-to-text (G2T) generation networks on HuggingFace Transformers 
4, partially based on previous works [KGPT](https://github.com/wenhuchen/KGPT), 
[PLM](https://github.com/UKPLab/plms-graph2text), and [JointGT](https://github.com/thu-coai/JointGT).

We include models of T5, BART, and JointGT for G2T generation, and we integrated FactSpotter into the inference to 
improve the factual faithfulness of generation.


## Dependencies

* Python 3.8+
* PyTorch 1.13
* PyTorch Scatter 2.0.9
* Transformers (Huggingface) 4.28

The code would encounter some bugs on Transformers >= 4.29.

* Numpy 1.20

In the code of the file g2t_gen_code/code/data.py at line 1120. In __getitem__ method of the WebNLGDataset class, np.int alias is used for instantiating a NumPy array. However, as stated in the NumPy doc, this alias is identical to the built-in int type and is no longer supported as of NumPy 1.20. If the version is higher than 1.20, the bug can be solved by replacing the "np.int" with "int". Thanks for the advice of [WilliamAboucaya](https://github.com/WilliamAboucaya). 

## Datasets

Our experiments contain several datasets, i.e., WebNLG, DART, SimpleQuestions and GrailQA.
They've been pre-processed to JSON format and put under the directory data/.

## Fine-tuning

T5 and BART model are able to be automatically loaded from HuggingFace library. For fine-tuning on JointGT, please
firstly download their checkpoint of pre-trained model 
([Google Drive](https://drive.google.com/drive/folders/1FGThWaTUs1cLvkd_GHCFV8mQEDW6qfIK?usp=sharing) /
[Tsinghua Cloud](https://cloud.tsinghua.edu.cn/d/79b009058cce484fa736/)), 
and fine-tune the pre-trained model on downstream datasets.

To train a certain model on a dataset, please execute the scripts provided in the folder `scripts`.
For instance,
```shell
bash scripts/grail/finetune_grail_plmt5base.sh
```

`--output_dir` is the directory to save the fine-tuning model. 

`--train_file` is directory for dataset.

`--predict_file` is the directory of val set.

`--model_path` is the pre-trained checkpoint used for fine-tuning. 

`--tokenizer_path` is the local directory of tokenizer (or on huggingface)

`--cls_model` is the type of model used by FactSpotter.

`--cls_model_path` is directory of trained FactSpotter weight. The weights for different datasets are available on 
[GoogleDrive](https://drive.google.com/drive/folders/1zsXmo2XPCmN60j90_BbIIs18l7DSI6PL?usp=sharing)

`--promotion_weight` is the weight for promoting factual generation. 
We set `promotion_weight=0` during the training of baseline models. 


## Inference

To infer a certain model on a dataset, please the scripts provided in the folder `scripts`. For instance,

```shell
bash scripts/grail/test_grail_plmt5base_cmp.sh
```
By default the code loads the models saved in `output_dir` if the provided directory starts with "out". 

During inference you can adjust the parameter `promotion_weight` to promote factual Graph-to-Text generation.

Our baseline models for different Graph-to-Text datasets are available on huggingface. 
If you want to infer with our provided weight, please keep `tokenizer_path` same as original model, 
but change the parameter `model_path` to be the provided huggingface model, such as

```shell
CUDA_VISIBLE_DEVICES=0 TRANSFORMERS_OFFLINE=1 python code/cli_gt.py \
        --do_predict \
        --model_path Inria-CEDAR/GrailQAT5S \
        --model_name t5-base \
        --output_dir out/grail_new_t5base \
        --train_file ../data/grail_new/train \
        --predict_file ../data/grail_new/iid \
        --tokenizer_path t5-base \
        --dataset grail_new \
        --predict_batch_size 1 \
        --append_another_bos \
        --max_input_length 512 \
        --max_output_length 256 \
        --num_beams 5 \
        --length_penalty 1. \
        --prefix grail_iid_t5base_beam5weight.15 \
        --cls_model "google/electra-small-discriminator" \
        --cls_model_path "electra_grail_fix.5pred.25ent.25both.1drop0.9670569557451876.pt" \
        --promotion_weight .15
```

| Datasets | Supported Models |
|-------|------|
| SimpleQ |[Inria-CEDAR/SimpleQBartL](https://huggingface.co/Inria-CEDAR/SimpleQBartL) <br>[Inria-CEDAR/SimpleQBartB](https://huggingface.co/Inria-CEDAR/SimpleQBartB) <br>[Inria-CEDAR/SimpleQT5JointGT](https://huggingface.co/Inria-CEDAR/SimpleQT5JointGT) <br>[Inria-CEDAR/SimpleQT5B](https://huggingface.co/Inria-CEDAR/SimpleQT5B) <br>[Inria-CEDAR/SimpleQT5S](https://huggingface.co/Inria-CEDAR/SimpleQT5S) |
| GrailQA |[Inria-CEDAR/GrailQABartL](https://huggingface.co/Inria-CEDAR/GrailQABartL) <br>[Inria-CEDAR/GrailQABartB](https://huggingface.co/Inria-CEDAR/GrailQABartB) <br>[Inria-CEDAR/GrailQAT5JointGT](https://huggingface.co/Inria-CEDAR/GrailQAT5JointGT) <br>[Inria-CEDAR/GrailQAT5B](https://huggingface.co/Inria-CEDAR/GrailQAT5B) <br>[Inria-CEDAR/GrailQAT5S](https://huggingface.co/Inria-CEDAR/GrailQAT5S) |
| WebNLG17Const |[Inria-CEDAR/WebNLG17ConstBartL](https://huggingface.co/Inria-CEDAR/WebNLG17ConstBartL) <br>[Inria-CEDAR/WebNLG17ConstBartB](https://huggingface.co/Inria-CEDAR/WebNLG17ConstBartB) <br>[Inria-CEDAR/WebNLG17ConstT5JointGT](https://huggingface.co/Inria-CEDAR/WebNLG17ConstT5JointGT) <br>[Inria-CEDAR/WebNLG17ConstT5B](https://huggingface.co/Inria-CEDAR/WebNLG17ConstT5B) <br>[Inria-CEDAR/WebNLG17ConstT5S](https://huggingface.co/Inria-CEDAR/WebNLG17ConstT5S) |
| WebNLG20 |[Inria-CEDAR/WebNLG20T5JointGT](https://huggingface.co/Inria-CEDAR/WebNLG20T5JointGT) <br>[Inria-CEDAR/WebNLG20BartL](https://huggingface.co/Inria-CEDAR/WebNLG20BartL) <br>[Inria-CEDAR/WebNLG20BartB](https://huggingface.co/Inria-CEDAR/WebNLG20BartB) <br>[Inria-CEDAR/WebNLG20T5B](https://huggingface.co/Inria-CEDAR/WebNLG20T5B) <br>[Inria-CEDAR/WebNLG20T5S](https://huggingface.co/Inria-CEDAR/WebNLG20T5S) |
|DART|[Inria-CEDAR/DartT5S](https://huggingface.co/Inria-CEDAR/DartT5S) <br>[Inria-CEDAR/DartT5B](https://huggingface.co/Inria-CEDAR/DartT5B) <br>[Inria-CEDAR/DartT5JointGT](https://huggingface.co/Inria-CEDAR/DartT5JointGT) <br>[Inria-CEDAR/DartBartB](https://huggingface.co/Inria-CEDAR/DartBartB) <br>[Inria-CEDAR/DartBartL](https://huggingface.co/Inria-CEDAR/DartBartL)|

## Thanks

Many thanks to the GitHub repositories of [Transformers](https://github.com/huggingface/transformers), [KGPT](https://github.com/wenhuchen/KGPT), [PLMGraph2Text](https://github.com/UKPLab/plms-graph2text), [JointGT](https://github.com/thu-coai/JointGT), and [BARTClosedBookQA](https://github.com/shmsw25/bart-closed-book-qa). 
Part of our codes are modified based on their codes.
