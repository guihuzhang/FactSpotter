import copy
import json
import random
import sys

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler

from coco_eval.pycocoevalcap.eval import COCOEvalCap
from coco_eval.pycocotools.coco import COCO


def run_coco_eval(data_ref, data_sys):
    """Run the COCO evaluator, return the resulting evaluation object (contains both
    system- and segment-level scores."""
    # convert references and system outputs to MS-COCO format in-memory
    coco_ref = create_coco_refs(data_ref)
    coco_sys = create_coco_sys(data_sys)

    print('Running MS-COCO evaluator...', file=sys.stderr)
    coco = COCO()
    coco.dataset = coco_ref
    coco.createIndex()

    coco_res = coco.loadRes(resData=coco_sys)
    coco_eval = COCOEvalCap(coco, coco_res)
    coco_eval.evaluate()

    return coco_eval


def create_coco_refs(data_ref):
    """Create MS-COCO human references JSON."""
    out = {'info': {}, 'licenses': [], 'images': [], 'type': 'captions', 'annotations': []}
    ref_id = 0
    for inst_id, refs in enumerate(data_ref):
        out['images'].append({'id': 'inst-%d' % inst_id})
        for ref in refs:
            out['annotations'].append({'image_id': 'inst-%d' % inst_id,
                                       'id': ref_id,
                                       'caption': ref})
            ref_id += 1
    return out


def create_coco_sys(data_sys):
    """Create MS-COCO system outputs JSON."""
    out = []
    for inst_id, inst in enumerate(data_sys):
        out.append({'image_id': 'inst-%d' % inst_id, 'caption': inst})
    return out


def safe_setting(matrix, x_start, x_end, y_start, y_end):
    if x_start >= matrix.shape[0] or y_start >= matrix.shape[0]:
        return

    matrix[x_start:min(matrix.shape[0], x_end), y_start:min(matrix.shape[1], y_end)] = 1
    return


# base dataset for KGPT
class KBDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_entity, max_fact, forbid_duplicate_relation=True, percent=1.0):
        super(KBDataset, self).__init__()
        with open(file_path, 'r') as f:
            self.data = json.load(f)

        if percent > 1:
            self.data = self.data[:int(percent)]
        else:
            selected_size = int(len(self.data) * percent)
            self.data = self.data[:selected_size]

        self.tokenizer = tokenizer
        self.max_entity = max_entity
        self.max_fact = max_fact
        self.forbid_duplicate_relation = forbid_duplicate_relation

    def linearize_v2(self, entity, lower_case=True):
        if lower_case:
            string = self.tokenizer.encode('[ENT] {}'.format(entity[0].lower()), add_special_tokens=False)
        else:
            string = self.tokenizer.encode('[ENT] {}'.format(entity[0]), add_special_tokens=False)
        triple_id = [1] * len(string)

        if lower_case:
            words = self.tokenizer.encode('[TRIPLE] [PRED] description [SUB] {} [TRIPLE]'.format(entity[1].lower()),
                                          add_special_tokens=False)
        else:
            words = self.tokenizer.encode('[TRIPLE] [PRED] description [SUB] {} [TRIPLE]'.format(entity[1]),
                                          add_special_tokens=False)

        string += words
        triple_id += [triple_id[-1] + 1] * len(words)

        added = set()
        for rel in entity[2]:
            if self.forbid_duplicate_relation and rel[0] in added:
                pass
            else:
                if lower_case:
                    words = self.tokenizer.encode('[PRED] {} [SUB] {} [TRIPLE]'.format(rel[0].lower(), rel[1].lower()),
                                                  add_special_tokens=False)
                else:
                    words = self.tokenizer.encode('[PRED] {} [SUB] {} [TRIPLE]'.format(rel[0], rel[1]),
                                                  add_special_tokens=False)
                string += words
                triple_id += [triple_id[-1] + 1] * len(words)
                added.add(rel[0])

            if len(added) >= self.max_fact:
                break

        return string, triple_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self):
        raise NotImplementedError


#  downstream dataset of KGPT
class DownStreamDataset(KBDataset):
    def __init__(self, file_path, tokenizer, max_entity, max_fact=12, max_enc_len=128, max_dec_len=30, encoder=None,
                 forbid_duplicate_relation=True, percent=1.0):
        super(DownStreamDataset, self).__init__(file_path, tokenizer, max_entity, max_fact, forbid_duplicate_relation,
                                                percent)

        print("Total samples = {}".format(len(self.data)))
        self.max_enc_len = max_enc_len
        self.max_dec_len = max_dec_len

        self.tokenizer = tokenizer
        self.pad_idx = tokenizer.pad_token_id
        self.encoder = encoder

    def get_reference(self, idx, do_lower_case=True):
        # change to COCO format
        if do_lower_case:
            return [_.strip(" ?.").lower() for _ in self.data[idx]['text']]
            # return [word_tokenize(_.strip(" ?.").lower()) for _ in self.data[idx]['text']]
        else:
            return [_.strip(" ?.") for _ in self.data[idx]['text']]
            # return [word_tokenize(_) for _ in self.data[idx]['text']]

    def get_entities(self, idx):
        return list(self.data[idx]['kbs'].keys())

    def __getitem__(self, idx):
        entry = self.data[idx]
        sentence = random.choice(entry['text']).strip(" ?.")
        KBs = entry['kbs']

        if self.encoder == 'sequence':
            strings = []
            entity_ids = []
            triple_ids = []

            for i, entity_label in enumerate(KBs):
                if i + 1 >= self.max_entity:
                    break

                entity = KBs[entity_label]

                string, triple_id = self.linearize_v2(entity)

                strings += string
                entity_ids += [i + 1] * len(string)
                triple_ids += triple_id

            position_ids = list(range(len(strings)))
            assert len(strings) == len(entity_ids) == len(triple_ids) == len(position_ids)

            if len(strings) >= self.max_enc_len:
                input_ids = torch.LongTensor(strings[:self.max_enc_len])
                entity_ids = torch.LongTensor(entity_ids[:self.max_enc_len])
                triple_ids = torch.LongTensor(triple_ids[:self.max_enc_len])
                position_ids = torch.LongTensor(position_ids[:self.max_enc_len])
            else:
                input_ids = torch.LongTensor(strings + [self.pad_idx] * (self.max_enc_len - len(strings)))
                entity_ids = torch.LongTensor(entity_ids + [0] * (self.max_enc_len - len(strings)))
                triple_ids = torch.LongTensor(triple_ids + [0] * (self.max_enc_len - len(strings)))
                position_ids = torch.LongTensor(position_ids + [0] * (self.max_enc_len - len(strings)))
            sentence = self.tokenizer.encode('[SOS] {} [EOS]'.format(sentence), add_special_tokens=False)
            #  modify here to debug
            # sentence = self.tokenizer.encode(' {}'.format(sentence), add_special_tokens=False)
            if len(sentence) >= self.max_dec_len:
                output_ids = torch.LongTensor(sentence[:self.max_dec_len])
            else:
                output_ids = torch.LongTensor(
                    sentence[:self.max_dec_len] + [self.pad_idx] * (self.max_dec_len - len(sentence)))

            return input_ids, entity_ids, triple_ids, position_ids, output_ids[:-1], output_ids[1:]

        elif self.encoder == 'graph':
            strings = []

            term_level_mask = torch.eye(self.max_enc_len).bool()
            triple_level_mask = torch.eye(self.max_enc_len).bool()
            entity_level_mask = torch.eye(self.max_enc_len).bool()
            fact_level_mask = torch.eye(self.max_enc_len).bool()
            gather_level_mask = torch.eye(self.max_enc_len).bool()

            offset = 0
            entity_positions = []
            for i, entity_label in enumerate(KBs):
                if i >= self.max_entity:
                    break

                entity = KBs[entity_label]

                string = self.tokenizer.encode('[ENT]')
                entity_positions.append(offset)
                offset += 1

                triple_positions = []
                entity[2].insert(0, ['description', entity[1]])
                added = set()
                for rel in entity[2]:
                    if offset >= self.max_enc_len - 1:
                        break
                    elif self.forbid_duplicate_relation and rel[0] in added:
                        continue
                    else:
                        words = self.tokenizer.encode('[TRIPLE]', add_special_tokens=False)
                        triple_positions.append(offset)
                        string += words
                        offset += len(words)

                        words = self.tokenizer.encode('{} [PRED] {} [SUB] {}'.format(entity[0], rel[0], rel[1]),
                                                      add_special_tokens=False)
                        string += words
                        safe_setting(term_level_mask, offset, offset + len(words), offset, offset + len(words))
                        offset += len(words)
                        safe_setting(triple_level_mask, triple_positions[-1], triple_positions[-1] + 1,
                                     triple_positions[-1] + 1, offset)

                        added.add(rel[0])

                    if len(added) >= self.max_fact:
                        break

                entity_level_mask[entity_positions[-1], triple_positions] = 1
                strings += string

                # Reserve at least 4 tokens for the next entity
                if offset >= self.max_enc_len - 5:
                    break

            for entity_position in entity_positions:
                fact_level_mask[entity_position, entity_positions] = 1

            for i in range(len(entity_positions) - 1):
                gather_level_mask[entity_positions[i] + 1: entity_positions[i + 1], entity_positions] = 1
            gather_level_mask[entity_positions[-1] + 1:offset, entity_positions] = 1

            if len(strings) >= self.max_enc_len:
                input_ids = torch.LongTensor(strings[:self.max_enc_len])
            else:
                input_ids = torch.LongTensor(strings + [self.pad_idx] * (self.max_enc_len - len(strings)))

            sentence = self.tokenizer.encode('[SOS] {} [EOS]'.format(sentence), add_special_tokens=False)
            if len(sentence) >= self.max_dec_len:
                output_ids = torch.LongTensor(sentence[:self.max_dec_len])
            else:
                output_ids = torch.LongTensor(
                    sentence[:self.max_dec_len] + [self.pad_idx] * (self.max_dec_len - len(sentence)))

            term_level_mask = ~term_level_mask
            triple_level_mask = ~triple_level_mask
            entity_level_mask = ~entity_level_mask
            fact_level_mask = ~fact_level_mask
            gather_level_mask = ~gather_level_mask

            return input_ids, term_level_mask, triple_level_mask, entity_level_mask, fact_level_mask, \
                   gather_level_mask, output_ids[:-1],


# Pre-training dataset (wikidata)
class WikidataDataset(Dataset):
    def __init__(self, logger, args, data_path, knowledge_file, tokenizer, mode):
        self.data_path = data_path
        self.tokenizer = tokenizer
        # Load data
        with open(self.data_path + '.json', 'r') as f:
            self.data = json.load(f)

        self.knowledge = knowledge_file

        print("Total samples = {}; Total entities = {}".format(len(self.data), len(self.knowledge)))

        if args.debug:
            self.data = self.data[:1000]
        assert type(self.data) == list
        assert all(["id" in d for d in self.data]), self.data[0].keys()
        if type(self.data[0]["id"]) == int:
            for i in range(len(self.data)):
                self.data[i]["id"] = str(self.data[i]["id"])

        self.args = args
        self.data_type = mode
        self.metric = "BLEU"
        self.forbid_duplicate_relation = True
        self.max_fact = 8
        self.max_entity = 12
        self.mask_prob = eval(args.mask_prob)

        # Get the ids for special tokens
        self.head_ids, self.rel_ids, self.tail_ids = self.tokenizer.encode(' [head]', add_special_tokens=False), \
                                                     self.tokenizer.encode(' [relation]', add_special_tokens=False), \
                                                     self.tokenizer.encode(' [tail]', add_special_tokens=False)
        self.graph_ids, self.text_ids = self.tokenizer.encode(' [graph]', add_special_tokens=False), \
                                        self.tokenizer.encode(' [text]', add_special_tokens=False)

        if "bart" in self.args.model_name:
            self.mask_token = self.tokenizer.mask_token
            self.mask_token_id = self.tokenizer.mask_token_id
        else:
            self.mask_token = self.tokenizer.additional_special_tokens[0]
            self.mask_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.additional_special_tokens[0])

        if "bart" in self.args.model_name:
            if self.args.append_another_bos:
                self.add_bos_id = [self.tokenizer.bos_token_id] * 2
            else:
                self.add_bos_id = [self.tokenizer.bos_token_id]
        else:
            self.add_bos_id = []

    def __len__(self):
        return len(self.data)

    def rel_mask(self, rel, text_relation, relation_change, p=0.15):
        # Mask relations
        # rel_input: relation ids in the encoder input (corrupted)
        # rel_input_token: relation tokens for rel_input
        # rel_label: relation ids in the encoder output (complete)
        # rel_label_token: relation tokens for rel_label

        rel_label = relation_change[rel][1]
        rel_label_token = copy.deepcopy(rel)

        if random.random() > p:
            rel_input = copy.deepcopy(rel_label)
            rel_input_token = copy.deepcopy(rel_label_token)
        else:
            # Similar to BERT, 80% mask, 10% replace, 10% unchanged
            step_prob = random.random()
            if step_prob < 0.8:
                rel_input = [self.mask_token_id] * len(rel_label)
                rel_input_token = ' '.join([self.mask_token] * len(rel_label))
            else:
                if step_prob < 0.9:
                    replace_rel = random.choice(text_relation)
                    replace_rel_ids = relation_change[replace_rel][1]
                    if len(rel_label) >= len(replace_rel_ids):
                        rel_input = replace_rel_ids + [self.mask_token_id] * (len(rel_label) - len(replace_rel_ids))
                        rel_input_token = ' '.join(
                            [replace_rel] + [self.mask_token] * (len(rel_label) - len(replace_rel_ids)))
                    else:
                        rel_input = replace_rel_ids[:len(rel_label)]
                        rel_input_token = ' '.join(self.tokenizer.convert_ids_to_tokens(rel_input))
                else:
                    rel_input = copy.deepcopy(rel_label)
                    rel_input_token = copy.deepcopy(rel_label_token)

        return rel_label, rel_label_token, rel_input, rel_input_token

    def linearize_v2(self, entity, entity_change, text_relation, head_ids, rel_ids, tail_ids,
                     relation_change, cnt_edge, adj_matrix):
        # Linearize knowledge graphs into sequences, modified based on
        # https://github.com/wenhuchen/KGPT/blob/main/code/DataLoader.py

        # string: encoder input ids (corrupted)
        # string_label: encoder label ids (complete)
        # string_tokens: encoder input tokens
        # string_label_tokens: encoder label tokens
        # nodes: node ids for each token
        # edges: edge ids for each token

        if len(entity[0]) == 0:
            return [], [], '', [], '', [], [], cnt_edge, adj_matrix
        nodes, edges = [], []
        string_label = copy.deepcopy(head_ids)
        string = copy.deepcopy(string_label)
        string_tokens = ' [head]'
        string_label_tokens = ' [head]'
        nodes.extend([-1] * len(string_label))
        edges.extend([-1] * len(string_label))

        string_label += entity_change[entity[0]][3]
        string_label_tokens += ' {}'.format(entity[0])
        string += entity_change[entity[0]][1]
        string_tokens += ' {}'.format(entity_change[entity[0]][0])
        nodes.extend([entity_change[entity[0]][4]] * len(entity_change[entity[0]][3]))
        edges.extend([-1] * len(entity_change[entity[0]][3]))

        triple_id = [1] * len(string)

        # Deal with the description relation in kgtext
        if len(entity[1]) != 0:
            rel_label, rel_label_token, rel_input, rel_input_token = self.rel_mask('description', text_relation,
                                                                                   relation_change, p=self.mask_prob[2])

            words = rel_ids + rel_input + tail_ids + entity_change[entity[1]][1]
            words_tokens = ' [relation] ' + rel_input_token + ' [tail] ' + entity_change[entity[1]][0]
            words_label = rel_ids + rel_label + tail_ids + entity_change[entity[1]][3]
            words_label_tokens = ' [relation] {} [tail] {}'.format(rel_label_token, entity[1])
            nodes.extend(
                [-1] * (len(rel_ids) + len(rel_label) + len(tail_ids)) + [entity_change[entity[1]][4]] * len(
                    entity_change[entity[1]][3]))
            edges.extend([-1] * len(rel_ids) + [cnt_edge] * len(rel_label) + [-1] * (
                    len(tail_ids) + len(entity_change[entity[1]][3])))
            if entity_change[entity[0]][4] < len(adj_matrix) and entity_change[entity[1]][4] < len(adj_matrix):
                adj_matrix[entity_change[entity[0]][4]][entity_change[entity[1]][4]] = cnt_edge

            cnt_edge += 1

            string += words
            string_tokens += words_tokens
            string_label += words_label
            string_label_tokens += words_label_tokens
            triple_id += [triple_id[-1] + 1] * len(words)

        added = set()
        # Deal with other triples in knowledge graphs
        for rel in entity[2]:
            if self.forbid_duplicate_relation and rel[0] in added:
                pass
            else:
                if len(rel[0]) != 0 and len(rel[1]) != 0:
                    rel_label, rel_label_token, rel_input, rel_input_token = self.rel_mask(
                        rel[0], text_relation, relation_change, p=self.mask_prob[2])
                    words_label = rel_ids + rel_label + tail_ids + entity_change[rel[1]][3]
                    words_label_tokens = ' [relation] {} [tail] {}'.format(rel_label_token, rel[1])
                    words = rel_ids + rel_input + tail_ids + entity_change[rel[1]][1]
                    words_tokens = ' [relation] ' + rel_input_token + ' [tail] ' + entity_change[rel[1]][0]
                    nodes.extend(
                        [-1] * (len(rel_ids) + len(rel_label) + len(tail_ids)) + [entity_change[rel[1]][4]] * len(
                            entity_change[rel[1]][3]))
                    edges.extend([-1] * len(rel_ids) + [cnt_edge] * len(rel_label) + [-1] * (
                            len(tail_ids) + len(entity_change[rel[1]][3])))
                    if entity_change[entity[0]][4] < len(adj_matrix) and entity_change[rel[1]][4] < len(adj_matrix):
                        adj_matrix[entity_change[entity[0]][4]][entity_change[rel[1]][4]] = cnt_edge

                    cnt_edge += 1

                    string += words
                    string_tokens += words_tokens
                    string_label += words_label
                    string_label_tokens += words_label_tokens
                    triple_id += [triple_id[-1] + 1] * len(words)
                    added.add(rel[0])

            if len(added) >= self.max_fact:
                break

        assert len(string) == len(string_label) == len(nodes) == len(edges)

        return string, triple_id, string_tokens, string_label, string_label_tokens, nodes, edges, cnt_edge, adj_matrix

    def get_all_entities_per_sample(self, mark_entity_number, mark_entity):
        text_entity = set()
        text_relation = set()
        for entity_id in mark_entity_number:
            entity = self.knowledge[entity_id]
            if len(entity[0]) == 0:
                continue
            if len(entity[1]) != 0:
                text_entity.add(entity[1])
                text_relation.add('description')

            added = set()
            for rel in entity[2]:
                if self.forbid_duplicate_relation and rel[0] in added:
                    pass
                else:
                    if len(rel[0]) != 0 and len(rel[1]) != 0:
                        text_relation.add(rel[0])
                        text_entity.add(rel[1])
                        added.add(rel[0])

                if len(added) >= self.max_fact:
                    break

        text_entity_list = list(text_entity)
        text_relation_list = list(text_relation)
        for entity_ele in mark_entity:
            if entity_ele in text_entity_list:
                text_entity_list.remove(entity_ele)

        return text_entity_list, text_relation_list

    # Acquire the result of each entity after perturbation
    def get_change_per_sample(self, mark_entity, text_entity, text_relation):
        ent_change = {}
        total_entity = mark_entity + text_entity
        # mark entity prob = 0.4 (mask_prob[0])
        mark_entity_mask = random.choices([0.0, 1.0], weights=[self.mask_prob[0], 1.0 - self.mask_prob[0]],
                                          k=len(mark_entity))
        # text entity prob = 0.2 (mask_prob[1])
        text_entity_mask = random.choices([0.0, 1.0], weights=[self.mask_prob[1], 1.0 - self.mask_prob[1]],
                                          k=len(text_entity))
        total_entity_mask = np.concatenate((mark_entity_mask, text_entity_mask))

        assert len(total_entity_mask) == len(total_entity)

        for ent_id in range(len(total_entity)):
            entity_toks = self.tokenizer.encode(" {}".format(total_entity[ent_id]), add_special_tokens=False)
            if total_entity_mask[ent_id] == 0:
                tmp_prob = random.random()
                if tmp_prob < 0.8:
                    token_str = " ".join([self.mask_token] * len(entity_toks))
                    token_id = [self.mask_token_id] * len(entity_toks)
                    ent_change[total_entity[ent_id]] = [token_str, token_id, total_entity[ent_id], entity_toks, ent_id]
                else:
                    if tmp_prob < 0.9:
                        replace_entity = random.choice(total_entity)
                        replace_entity_toks = self.tokenizer.encode(" {}".format(replace_entity),
                                                                    add_special_tokens=False)
                        if len(entity_toks) > len(replace_entity_toks):
                            token_str = "{} {}".format(replace_entity, " ".join(
                                [self.mask_token] * (len(entity_toks) - len(replace_entity_toks))))
                            token_id = replace_entity_toks + [self.mask_token_id] * (
                                    len(entity_toks) - len(replace_entity_toks))
                            ent_change[total_entity[ent_id]] = [token_str, token_id, total_entity[ent_id], entity_toks,
                                                                ent_id]
                        else:
                            token_id = replace_entity_toks[:len(entity_toks)]
                            token_str = " ".join(self.tokenizer.convert_ids_to_tokens(token_id))
                            ent_change[total_entity[ent_id]] = [token_str, token_id, total_entity[ent_id], entity_toks,
                                                                ent_id]
                    else:
                        ent_change[total_entity[ent_id]] = [total_entity[ent_id], entity_toks, total_entity[ent_id],
                                                            entity_toks, ent_id]
            else:
                ent_change[total_entity[ent_id]] = [total_entity[ent_id], entity_toks, total_entity[ent_id],
                                                    entity_toks, ent_id]

        # relation change only includes the relation tokens and ids
        rel_change = {}
        for rel_id in range(len(text_relation)):
            rel_change[text_relation[rel_id]] = [text_relation[rel_id],
                                                 self.tokenizer.encode(' {}'.format(text_relation[rel_id]),
                                                                       add_special_tokens=False)]

        return ent_change, rel_change

    # Acquire the masked text
    def text_mask(self, word, word_cnt, p=0.15):
        if len(word) == 0:
            return [], '', [], '', []

        word_label = self.tokenizer.encode(" {}".format(word), add_special_tokens=False)
        word_whole = [word_cnt] * len(word_label)
        word_label_token = copy.deepcopy(word)

        if random.random() > p:
            word_input = copy.deepcopy(word_label)
            word_input_token = copy.deepcopy(word_label_token)
        else:
            step_prob = random.random()
            if step_prob < 0.8:
                word_input = [self.mask_token_id] * len(word_label)
                word_input_token = ' '.join([self.mask_token] * len(word_label))
            else:
                if step_prob < 0.9:
                    replace_word = [random.randint(0, self.tokenizer.vocab_size - 1)]
                    replace_word_token = [self.tokenizer.convert_ids_to_tokens(replace_word[0])]
                    if len(word_label) >= len(replace_word):
                        word_input = replace_word + [self.mask_token_id] * (len(word_label) - len(replace_word))
                        word_input_token = ' '.join(
                            replace_word_token + [self.mask_token] * (len(word_label) - len(replace_word)))
                    else:
                        word_input = replace_word[:len(word_label)]
                        word_input_token = ' '.join(replace_word_token[:len(word_label)])
                else:
                    word_input = copy.deepcopy(word_label)
                    word_input_token = copy.deepcopy(word_label_token)

        return word_label, word_label_token, word_input, word_input_token, word_whole

    def truncate_pair_ar(self, a, b, add_bos_id, graph_ids, text_ids, node_ids, edge_ids):
        # add_bos_id + graph_ids + a + text_ids + b + eos_token_id
        length_a_b = self.args.max_input_length - len(add_bos_id) - len(graph_ids) - len(text_ids) - 1
        if len(a) + len(b) > length_a_b:
            a = a[:(length_a_b - len(b))]
            node_ids = node_ids[:(length_a_b - len(b))]
            edge_ids = edge_ids[:(length_a_b - len(b))]
        input_ids = add_bos_id + graph_ids + a + text_ids + b + [self.tokenizer.eos_token_id]
        input_node_ids = [-1] * (len(add_bos_id) + len(graph_ids)) + node_ids + [-1] * (len(text_ids) + len(b) + 1)
        input_edge_ids = [-1] * (len(add_bos_id) + len(graph_ids)) + edge_ids + [-1] * (len(text_ids) + len(b) + 1)
        attn_mask = [1] * len(input_ids) + [0] * (self.args.max_input_length - len(input_ids))
        input_ids += [self.tokenizer.pad_token_id] * (self.args.max_input_length - len(input_ids))
        input_node_ids += [-1] * (self.args.max_input_length - len(input_node_ids))
        input_edge_ids += [-1] * (self.args.max_input_length - len(input_edge_ids))
        assert len(input_ids) == len(attn_mask) == self.args.max_input_length == len(input_node_ids) == len(
            input_edge_ids)
        return input_ids, attn_mask, input_node_ids, input_edge_ids

    def truncate_pair_ae(self, a, b, a_ori, add_bos_id, graph_ids, text_ids, node_ids, edge_ids):
        # add_bos_id + graph_ids + a + text_ids + b + eos_token_id
        length_a_b = self.args.max_input_length - len(add_bos_id) - len(graph_ids) - len(text_ids) - 1
        if len(a) + len(b) > length_a_b:
            a = a[:(length_a_b - len(b))]
            node_ids = node_ids[:(length_a_b - len(b))]
            edge_ids = edge_ids[:(length_a_b - len(b))]
            a_ori = a_ori[:(length_a_b - len(b))]
        input_ids = add_bos_id + graph_ids + a + text_ids + b + [self.tokenizer.eos_token_id]
        input_ids_ori = add_bos_id + graph_ids + a_ori + text_ids + b + [self.tokenizer.eos_token_id]
        input_node_ids = [-1] * (len(add_bos_id) + len(graph_ids)) + node_ids + [-1] * (len(text_ids) + len(b) + 1)
        input_edge_ids = [-1] * (len(add_bos_id) + len(graph_ids)) + edge_ids + [-1] * (len(text_ids) + len(b) + 1)
        attn_mask = [1] * len(input_ids) + [0] * (self.args.max_input_length - len(input_ids))
        input_ids += [self.tokenizer.pad_token_id] * (self.args.max_input_length - len(input_ids))
        input_ids_ori += [self.tokenizer.pad_token_id] * (self.args.max_input_length - len(input_ids_ori))
        input_node_ids += [-1] * (self.args.max_input_length - len(input_node_ids))
        input_edge_ids += [-1] * (self.args.max_input_length - len(input_edge_ids))
        assert len(input_ids) == len(attn_mask) == len(input_node_ids) == len(input_edge_ids)
        assert len(input_ids) == len(input_ids_ori) == self.args.max_input_length
        return input_ids, attn_mask, input_ids_ori, input_node_ids, input_edge_ids

    def truncate_pair_ot(self, a, add_bos_id, graph_ids, text_ids, node_ids, edge_ids):
        # add_bos_id + graph_ids + a + text_ids + eos_token_id
        length_a_b = self.args.max_input_length - len(add_bos_id) - len(graph_ids) - len(text_ids) - 1
        if len(a) > length_a_b:
            a = a[:length_a_b]
            node_ids = node_ids[:length_a_b]
            edge_ids = edge_ids[:length_a_b]
        # empty text
        input_ids = add_bos_id + graph_ids + a + text_ids + [self.tokenizer.eos_token_id]
        input_node_ids = [-1] * (len(add_bos_id) + len(graph_ids)) + node_ids + [-1] * (len(text_ids) + 1)
        input_edge_ids = [-1] * (len(add_bos_id) + len(graph_ids)) + edge_ids + [-1] * (len(text_ids) + 1)
        attn_mask = [1] * len(input_ids) + [0] * (self.args.max_input_length - len(input_ids))
        input_ids += [self.tokenizer.pad_token_id] * (self.args.max_input_length - len(input_ids))
        input_node_ids += [-1] * (self.args.max_input_length - len(input_node_ids))
        input_edge_ids += [-1] * (self.args.max_input_length - len(input_edge_ids))
        assert len(input_ids) == len(attn_mask) == self.args.max_input_length == len(input_node_ids) == len(
            input_edge_ids)
        return input_ids, attn_mask, input_node_ids, input_edge_ids

    # Prepare data for text reconstruction
    def ar_prep_data(self, answers_input, answers, questions, add_bos_id, graph_ids, text_ids, node_ids, edge_ids):
        # merge mask in answers_input
        text_pertubed_input = []
        for data_id in range(len(answers_input)):
            if len(text_pertubed_input) == 0:
                text_pertubed_input.append(answers_input[data_id])
            else:
                if answers_input[data_id] != self.mask_token_id:
                    text_pertubed_input.append(answers_input[data_id])
                else:
                    if text_pertubed_input[-1] != self.mask_token_id:
                        text_pertubed_input.append(answers_input[data_id])

        # add bos and eos
        decoder_label_ids = copy.deepcopy(answers)
        if len(decoder_label_ids) > self.args.max_output_length - len(add_bos_id) - 1:
            decoder_label_ids = decoder_label_ids[:(self.args.max_output_length - len(add_bos_id) - 1)]
        decoder_label_ids = add_bos_id + decoder_label_ids + [self.tokenizer.eos_token_id]
        decoder_attn_mask = [1] * len(decoder_label_ids) + [0] * (self.args.max_output_length - len(decoder_label_ids))
        decoder_label_ids += [self.tokenizer.pad_token_id] * (self.args.max_output_length - len(decoder_label_ids))
        # decoder_label_ids += [self.tokenizer.eos_token_id] * (self.args.max_output_length - len(decoder_label_ids))
        assert len(decoder_label_ids) == self.args.max_output_length
        assert len(decoder_label_ids) == len(decoder_attn_mask)

        input_ids, input_attn_mask, input_node_ids, input_edge_ids = self.truncate_pair_ar(questions,
                                                                                           text_pertubed_input,
                                                                                           add_bos_id, graph_ids,
                                                                                           text_ids, node_ids, edge_ids)

        return input_ids, input_attn_mask, decoder_label_ids, decoder_attn_mask, input_node_ids, input_edge_ids

    # Prepare data for graph reconstruction
    def ae_prep_data(self, questions_input, questions, answers, add_bos_id, graph_ids, text_ids, node_ids, edge_ids):

        input_ids, attn_mask, input_ids_ori, input_node_ids, input_edge_ids = self.truncate_pair_ae(questions_input,
                                                                                                    answers, questions,
                                                                                                    add_bos_id,
                                                                                                    graph_ids, text_ids,
                                                                                                    node_ids, edge_ids)

        encoder_label_ids = [-1 if input_ids[idx] == input_ids_ori[idx] else input_ids_ori[idx] for idx in
                             range(len(input_ids))]

        assert len(encoder_label_ids) == len(input_ids)

        return input_ids, attn_mask, encoder_label_ids, input_node_ids, input_edge_ids

    # Prepare data for embedding alignment
    def ot_prep_data(self, answers, questions, add_bos_id, graph_ids, text_ids, node_ids, edge_ids, word_whole_ids):
        # add bos and eos
        decoder_label_ids = copy.deepcopy(answers)
        if len(decoder_label_ids) > self.args.max_output_length - len(add_bos_id) - 1:
            decoder_label_ids = decoder_label_ids[:(self.args.max_output_length - len(add_bos_id) - 1)]
            word_whole_ids = word_whole_ids[:(self.args.max_output_length - len(add_bos_id) - 1)]
        decoder_label_ids = add_bos_id + decoder_label_ids + [self.tokenizer.eos_token_id]
        decoder_word_whole_ids = [-1] * len(add_bos_id) + word_whole_ids + [-1]
        decoder_attn_mask = [1] * len(decoder_label_ids) + [0] * (self.args.max_output_length - len(decoder_label_ids))
        decoder_label_ids += [self.tokenizer.pad_token_id] * (self.args.max_output_length - len(decoder_label_ids))
        # decoder_label_ids += [self.tokenizer.eos_token_id] * (self.args.max_output_length - len(decoder_label_ids))
        decoder_word_whole_ids += [-1] * (self.args.max_output_length - len(decoder_word_whole_ids))
        assert len(decoder_label_ids) == self.args.max_output_length
        assert len(decoder_label_ids) == len(decoder_attn_mask) == len(decoder_word_whole_ids)

        input_ids, input_attn_mask, input_node_ids, input_edge_ids = self.truncate_pair_ot(questions, add_bos_id,
                                                                                           graph_ids, text_ids,
                                                                                           node_ids, edge_ids)

        return input_ids, input_attn_mask, decoder_label_ids, decoder_attn_mask, input_node_ids, input_edge_ids, decoder_word_whole_ids

    def __getitem__(self, idx):

        entry = self.data[idx]

        entities = []
        for _ in entry['kblinks']:
            if _ is not None and _ in self.knowledge and _ not in entities:
                entities.append(_)

        # strings / string tokens: corrupted linearized graph (ids / tokens)
        # strings_label / string_label_tokens: complete linearized graph (ids / tokens)
        # entity_ids: entity index of the corrupted linearized graph
        # node_ids / edge_ids: node / edge index of the corrupted linearized graph

        strings = []
        strings_label = []
        entity_ids = []
        triple_ids = []
        node_ids = []
        edge_ids = []
        strings_tokens = ''
        strings_label_tokens = ''

        # mark_entity: entities with KB numbers which are important for this task
        # text_entity: entities without KB numbers but only with text, which are less important
        mark_entity = [self.knowledge[ele_entity][0] for ele_entity in entities] + [
            self.knowledge[entry['title_kb_id']][0]]
        mark_entity_number = entities + [entry['title_kb_id']]
        text_entity, text_relation = self.get_all_entities_per_sample(mark_entity_number, mark_entity)
        entity_change, relation_change = self.get_change_per_sample(mark_entity, text_entity, text_relation)
        total_entity = mark_entity + text_entity
        adj_matrix = [[-1] * (self.args.max_node_length + 1) for _ in range(self.args.max_node_length + 1)]
        cnt_edge = 0

        if 'title' in entry:
            entity = self.knowledge[entry['title_kb_id']]

            string, triple_id, string_tokens, string_label, string_label_tokens, nodes, edges, cnt_edge, adj_matrix = \
                self.linearize_v2(
                    entity,
                    entity_change,
                    text_relation,
                    self.head_ids,
                    self.rel_ids, self.tail_ids,
                    relation_change, cnt_edge, adj_matrix)

            strings += string
            strings_tokens += string_tokens
            strings_label += string_label
            strings_label_tokens += string_label_tokens
            entity_ids += [0] * len(string)
            triple_ids += triple_id
            node_ids += nodes
            edge_ids += edges

        for i, entity_id in enumerate(entities):
            if i + 1 >= self.max_entity:
                break

            entity = self.knowledge[entity_id]

            string, triple_id, string_tokens, string_label, string_label_tokens, nodes, edges, cnt_edge, adj_matrix = self.linearize_v2(
                entity,
                entity_change,
                text_relation,
                self.head_ids,
                self.rel_ids, self.tail_ids,
                relation_change, cnt_edge, adj_matrix)

            strings += string
            strings_tokens += string_tokens
            strings_label += string_label
            strings_label_tokens += string_label_tokens
            entity_ids += [i + 1] * len(string)
            triple_ids += triple_id
            node_ids += nodes
            edge_ids += edges

        position_ids = list(range(len(strings)))
        assert len(strings) == len(entity_ids) == len(triple_ids) == len(position_ids)
        assert len(strings) == len(strings_label) == len(node_ids) == len(edge_ids)

        # words_input_ids / words_input_tokens: corrupted texts (ids / tokens)
        # words_label_ids / words_label_tokens: complete texts (ids / tokens)
        # words_whole_ids: word index

        words_label_ids, words_label_tokens, words_input_ids, words_input_tokens, words_whole_ids = [], '', [], '', []

        word_cnt = 0
        for word in entry['text']:
            if word not in total_entity:
                word_label_ids, word_label_tokens, word_input_ids, word_input_tokens, word_whole_ids = \
                    self.text_mask(word, word_cnt, p=self.mask_prob[4])
            else:
                word_label_ids, word_label_tokens, word_input_ids, word_input_tokens, word_whole_ids = \
                    self.text_mask(word, word_cnt, p=self.mask_prob[3])

            words_label_ids += word_label_ids
            words_label_tokens += ' ' + word_label_tokens
            words_input_ids += word_input_ids
            words_input_tokens += ' ' + word_input_tokens
            words_whole_ids += word_whole_ids
            if len(word_label_ids) > 0:
                word_cnt += 1

        assert len(words_input_ids) == len(words_label_ids) == len(words_whole_ids)

        input_ids_ar, attn_mask_ar, decoder_label_ids, decoder_attn_mask, input_node_ids_ar, input_edge_ids_ar = \
            self.ar_prep_data(words_input_ids, words_label_ids, strings_label, self.add_bos_id, self.graph_ids,
                              self.text_ids, node_ids, edge_ids)
        input_ids_ae, attn_mask_ae, encoder_label_ids, input_node_ids_ae, input_edge_ids_ae = \
            self.ae_prep_data(strings, strings_label, words_label_ids, self.add_bos_id, self.graph_ids,
                              self.text_ids, node_ids, edge_ids)
        input_ids_ot, attn_mask_ot, decoder_label_ids_ot, decoder_attn_mask_ot, input_node_ids_ot, input_edge_ids_ot, decoder_whole_ids_ot = \
            self.ot_prep_data(words_label_ids, strings_label, self.add_bos_id, self.graph_ids,
                              self.text_ids, node_ids, edge_ids, words_whole_ids)

        node_length_ar, node_length_ae, node_length_ot = max(input_node_ids_ar) + 1, max(input_node_ids_ae) + 1, max(
            input_node_ids_ot) + 1
        edge_length_ar, edge_length_ae, edge_length_ot = max(input_edge_ids_ar) + 1, max(input_edge_ids_ae) + 1, max(
            input_edge_ids_ot) + 1
        word_length_ot = max(decoder_whole_ids_ot) + 1

        def masked_fill(src, masked_value, fill_value):
            return [src[src_id] if src[src_id] != masked_value and src[src_id] < fill_value else fill_value for src_id
                    in range(len(src))]

        input_node_ids_ar, input_edge_ids_ar = masked_fill(input_node_ids_ar, -1, self.args.max_node_length), \
                                               masked_fill(input_edge_ids_ar, -1, self.args.max_edge_length)
        input_node_ids_ae, input_edge_ids_ae = masked_fill(input_node_ids_ae, -1, self.args.max_node_length), \
                                               masked_fill(input_edge_ids_ae, -1, self.args.max_edge_length)
        input_node_ids_ot, input_edge_ids_ot = masked_fill(input_node_ids_ot, -1, self.args.max_node_length), \
                                               masked_fill(input_edge_ids_ot, -1, self.args.max_edge_length)
        decoder_whole_ids_ot = masked_fill(decoder_whole_ids_ot, -1, self.args.max_output_length)

        def masked_fill_matrix(adj_matrix_input, masked_value, fill_value):
            adj_matrix_tmp = copy.deepcopy(adj_matrix_input)
            for a_id in range(len(adj_matrix_tmp)):
                for b_id in range(len(adj_matrix_tmp)):
                    if adj_matrix_tmp[a_id][b_id] == masked_value or adj_matrix_tmp[a_id][b_id] > fill_value:
                        adj_matrix_tmp[a_id][b_id] = fill_value
            return adj_matrix_tmp

        adj_matrix_ar = masked_fill_matrix(adj_matrix, -1, self.args.max_edge_length)
        adj_matrix_ae = masked_fill_matrix(adj_matrix, -1, self.args.max_edge_length)
        adj_matrix_ot = masked_fill_matrix(adj_matrix, -1, self.args.max_edge_length)

        assert len(input_ids_ar) == len(attn_mask_ar) == self.args.max_input_length == len(input_node_ids_ar) == len(
            input_edge_ids_ar)
        assert len(decoder_label_ids) == len(decoder_attn_mask) == self.args.max_output_length
        assert len(input_ids_ae) == len(attn_mask_ae) == len(encoder_label_ids) == self.args.max_input_length == len(
            input_node_ids_ae) == len(input_edge_ids_ae)
        assert len(input_ids_ot) == len(attn_mask_ot) == self.args.max_input_length == len(input_node_ids_ot) == len(
            input_edge_ids_ot)
        assert len(decoder_label_ids_ot) == len(decoder_attn_mask_ot) == self.args.max_output_length == len(
            decoder_whole_ids_ot)

        input_ids_ar = torch.LongTensor(input_ids_ar)
        attn_mask_ar = torch.LongTensor(attn_mask_ar)
        decoder_label_ids = torch.LongTensor(decoder_label_ids)
        decoder_attn_mask = torch.LongTensor(decoder_attn_mask)
        input_ids_ae = torch.LongTensor(input_ids_ae)
        attn_mask_ae = torch.LongTensor(attn_mask_ae)
        encoder_label_ids = torch.LongTensor(encoder_label_ids)
        input_node_ids_ar = torch.LongTensor(input_node_ids_ar)
        input_edge_ids_ar = torch.LongTensor(input_edge_ids_ar)
        input_node_ids_ae = torch.LongTensor(input_node_ids_ae)
        input_edge_ids_ae = torch.LongTensor(input_edge_ids_ae)
        node_length_ar = torch.LongTensor([node_length_ar])
        node_length_ae = torch.LongTensor([node_length_ae])
        edge_length_ar = torch.LongTensor([edge_length_ar])
        edge_length_ae = torch.LongTensor([edge_length_ae])
        adj_matrix_ar = torch.LongTensor(adj_matrix_ar)
        adj_matrix_ae = torch.LongTensor(adj_matrix_ae)

        input_ids_ot = torch.LongTensor(input_ids_ot)
        attn_mask_ot = torch.LongTensor(attn_mask_ot)
        input_node_ids_ot = torch.LongTensor(input_node_ids_ot)
        input_edge_ids_ot = torch.LongTensor(input_edge_ids_ot)
        decoder_label_ids_ot = torch.LongTensor(decoder_label_ids_ot)
        decoder_attn_mask_ot = torch.LongTensor(decoder_attn_mask_ot)
        decoder_whole_ids_ot = torch.LongTensor(decoder_whole_ids_ot)
        adj_matrix_ot = torch.LongTensor(adj_matrix_ot)
        word_length_ot = torch.LongTensor([word_length_ot])
        node_length_ot = torch.LongTensor([node_length_ot])
        edge_length_ot = torch.LongTensor([edge_length_ot])

        return input_ids_ar, attn_mask_ar, decoder_label_ids, decoder_attn_mask, input_node_ids_ar, \
               input_edge_ids_ar, node_length_ar, edge_length_ar, adj_matrix_ar, \
               input_ids_ae, attn_mask_ae, encoder_label_ids, input_node_ids_ae, input_edge_ids_ae, \
               node_length_ae, edge_length_ae, adj_matrix_ae, \
               input_ids_ot, attn_mask_ot, decoder_label_ids_ot, decoder_attn_mask_ot, decoder_whole_ids_ot, \
               input_node_ids_ot, input_edge_ids_ot, node_length_ot, edge_length_ot, word_length_ot, adj_matrix_ot


class WikidataDataLoader(DataLoader):

    def __init__(self, args, dataset, mode):
        if mode == "train":
            sampler = RandomSampler(dataset)
            batch_size = args.train_batch_size
        else:
            sampler = SequentialSampler(dataset)
            batch_size = args.predict_batch_size
        super(WikidataDataLoader, self).__init__(dataset, sampler=sampler, batch_size=batch_size,
                                                 num_workers=args.num_workers)


class WebNLGDataLoader(DataLoader):

    def __init__(self, args, dataset, mode):
        if mode == "train":
            sampler = RandomSampler(dataset)
            batch_size = args.train_batch_size
        else:
            sampler = SequentialSampler(dataset)
            batch_size = args.predict_batch_size
        super(WebNLGDataLoader, self).__init__(dataset, sampler=sampler, batch_size=batch_size,
                                               num_workers=args.num_workers)


# Downstream dataset (webnlg17, webquestions, pathquestions)
# Most parts are similar to WikidataDataset
class WebNLGDataset(Dataset):
    def __init__(self, logger, args, data_path, tokenizer, mode):
        self.data_path = data_path
        self.tokenizer = tokenizer
        with open(self.data_path + '.json', 'r') as f:
            self.data = json.load(f)
        print("Total samples = {}".format(len(self.data)))

        if args.debug:
            self.data = self.data[:1000]
        assert type(self.data) == list
        assert all(["id" in d for d in self.data]), self.data[0].keys()
        if type(self.data[0]["id"]) == int:
            for i in range(len(self.data)):
                self.data[i]["id"] = str(self.data[i]["id"])

        self.args = args
        self.data_type = mode

        self.metric = "BLEU"

        self.head_ids, self.rel_ids, self.tail_ids = self.tokenizer.encode(' [head]', add_special_tokens=False), \
                                                     self.tokenizer.encode(' [relation]', add_special_tokens=False), \
                                                     self.tokenizer.encode(' [tail]', add_special_tokens=False)
        self.graph_ids, self.text_ids = self.tokenizer.encode(' [graph]', add_special_tokens=False), \
                                        self.tokenizer.encode(' [text]', add_special_tokens=False)

        if "bart" in self.args.model_name:
            self.mask_token = self.tokenizer.mask_token
            self.mask_token_id = self.tokenizer.mask_token_id
        else:
            self.mask_token = self.tokenizer.additional_special_tokens[0]
            self.mask_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.additional_special_tokens[0])

        if "bart" in self.args.model_name:
            if self.args.append_another_bos:
                self.add_bos_id = [self.tokenizer.bos_token_id] * 2
            else:
                self.add_bos_id = [self.tokenizer.bos_token_id]
        else:
            self.add_bos_id = []
        self.prevent_predicates = ["class name", "entity name", "domains", "literal name", "sentence category",
                                   "Sentence Data Source", ]
        self.property_predicates = ["class name", "entity name", "literal name"]

    def __len__(self):
        return len(self.data)

    @staticmethod
    def linearize_v2(entity, entity_change, head_ids, rel_ids, tail_ids,
                     relation_change, cnt_edge, adj_matrix):
        # string_label: encoder ids
        # string_label_tokens: encoder tokens

        if len(entity[0]) == 0:
            return [], '', [], [], cnt_edge, adj_matrix
        nodes, edges = [], []
        string_label = copy.deepcopy(head_ids)
        string_label_tokens = ' [head]'
        nodes.extend([-1] * len(string_label))
        edges.extend([-1] * len(string_label))

        string_label += entity_change[entity[0]][0]
        string_label_tokens += ' {}'.format(entity[0])
        nodes.extend([entity_change[entity[0]][1]] * len(entity_change[entity[0]][0]))
        edges.extend([-1] * len(entity_change[entity[0]][0]))

        for rel in entity[2]:
            if len(rel[0]) != 0 and len(rel[1]) != 0:
                rel_label = relation_change[rel[0]]
                rel_label_token = copy.deepcopy(rel[0])
                words_label = rel_ids + rel_label + tail_ids + entity_change[rel[1]][0]
                words_label_tokens = ' [relation] {} [tail] {}'.format(rel_label_token, rel[1])
                nodes.extend(
                    [-1] * (len(rel_ids) + len(rel_label) + len(tail_ids)) + [entity_change[rel[1]][1]] * len(
                        entity_change[rel[1]][0]))
                edges.extend([-1] * len(rel_ids) + [cnt_edge] * len(rel_label) + [-1] * (
                        len(tail_ids) + len(entity_change[rel[1]][0])))
                if entity_change[entity[0]][1] < len(adj_matrix) and entity_change[rel[1]][1] < len(adj_matrix):
                    adj_matrix[entity_change[entity[0]][1]][entity_change[rel[1]][1]] = cnt_edge

                cnt_edge += 1
                string_label += words_label
                string_label_tokens += words_label_tokens
        assert len(string_label) == len(nodes) == len(edges)
        return string_label, string_label_tokens, nodes, edges, cnt_edge, adj_matrix

    @staticmethod
    def get_all_entities_per_sample(mark_entity_number, mark_entity, entry):
        text_entity = set()
        text_relation = set()
        for entity_id in mark_entity_number:
            entity = entry['kbs'][entity_id]
            if len(entity[0]) == 0:
                continue
            for rel in entity[2]:
                if len(rel[0]) != 0 and len(rel[1]) != 0:
                    text_relation.add(rel[0])
                    text_entity.add(rel[1])

        text_entity_list = list(text_entity)
        text_relation_list = list(text_relation)
        for entity_ele in mark_entity:
            if entity_ele in text_entity_list:
                text_entity_list.remove(entity_ele)

        return text_entity_list, text_relation_list

    def get_change_per_sample(self, mark_entity, text_entity, text_relation):
        # during fine-tuning, we don't mask entities or relations
        ent_change = {}
        total_entity = mark_entity + text_entity

        for ent_id in range(len(total_entity)):
            entity_toks = self.tokenizer.encode(" {}".format(total_entity[ent_id]), add_special_tokens=False)
            ent_change[total_entity[ent_id]] = [entity_toks, ent_id]

        # relation change only includes the relation tokens and ids
        rel_change = {}
        for rel_id in range(len(text_relation)):
            rel_change[text_relation[rel_id]] = self.tokenizer.encode(' {}'.format(text_relation[rel_id]),
                                                                      add_special_tokens=False)

        return ent_change, rel_change

    def truncate_pair_ar(self, a, add_bos_id, graph_ids, text_ids, node_ids, edge_ids):
        # add_bos_id + graph_ids + a + text_ids + b + eos_token_id
        length_a_b = self.args.max_input_length - len(add_bos_id) - len(graph_ids) - len(text_ids) - 1
        if len(a) > length_a_b:
            a = a[:length_a_b]
            node_ids = node_ids[:length_a_b]
            edge_ids = edge_ids[:length_a_b]
        input_ids = add_bos_id + graph_ids + a + text_ids + [self.tokenizer.eos_token_id]
        input_node_ids = [-1] * (len(add_bos_id) + len(graph_ids)) + node_ids + [-1] * (len(text_ids) + 1)
        input_edge_ids = [-1] * (len(add_bos_id) + len(graph_ids)) + edge_ids + [-1] * (len(text_ids) + 1)
        attn_mask = [1] * len(input_ids) + [0] * (self.args.max_input_length - len(input_ids))
        input_ids += [self.tokenizer.pad_token_id] * (self.args.max_input_length - len(input_ids))
        input_node_ids += [-1] * (self.args.max_input_length - len(input_node_ids))
        input_edge_ids += [-1] * (self.args.max_input_length - len(input_edge_ids))
        assert len(input_ids) == len(attn_mask) == self.args.max_input_length == len(input_node_ids) == len(
            input_edge_ids)
        return input_ids, attn_mask, input_node_ids, input_edge_ids

    def ar_prep_data(self, answers, questions, add_bos_id, graph_ids, text_ids, node_ids, edge_ids):
        # add bos and eos
        decoder_label_ids = copy.deepcopy(answers)
        if len(decoder_label_ids) > self.args.max_output_length - len(add_bos_id) - 1:
            decoder_label_ids = decoder_label_ids[:(self.args.max_output_length - len(add_bos_id) - 1)]
        decoder_label_ids = add_bos_id + decoder_label_ids + [self.tokenizer.eos_token_id]
        decoder_attn_mask = [1] * len(decoder_label_ids) + [0] * (self.args.max_output_length - len(decoder_label_ids))
        decoder_label_ids += [self.tokenizer.pad_token_id] * (self.args.max_output_length - len(decoder_label_ids))
        # decoder_label_ids += [self.tokenizer.eos_token_id] * (self.args.max_output_length - len(decoder_label_ids))
        assert len(decoder_label_ids) == self.args.max_output_length == len(decoder_attn_mask)

        input_ids, input_attn_mask, input_node_ids, input_edge_ids = self.truncate_pair_ar(questions, add_bos_id,
                                                                                           graph_ids, text_ids,
                                                                                           node_ids, edge_ids)

        return input_ids, input_attn_mask, decoder_label_ids, decoder_attn_mask, input_node_ids, input_edge_ids

    def __getitem__(self, idx):

        entry = self.data[idx]

        entities = []
        for _ in entry['kbs']:
            entities.append(_)

        #  dataloader for entity names
        entity_names = set()
        entity_constrain_len = 0
        for each_triple in entry['kbs']:
            for each_po in entry['kbs'][each_triple][2]:
                if " : " in each_po[1] and 'entity name' == each_po[0]:
                    p_and_o = each_po[1].split(" : ")
                    entity_names.add(p_and_o[0].lower())
        entity_constrain_list = []
        if self.data_type != "train":
            for each in entity_names:
                entity_constrain_list.append(self.tokenizer.encode("" + each, add_special_tokens=False))
                con_len = len(entity_constrain_list[-1])
                if con_len > entity_constrain_len:
                    entity_constrain_len = con_len
        np_constrain_entity = np.zeros([len(entity_constrain_list), entity_constrain_len], dtype=np.int)
        for idx in range(0, len(entity_constrain_list)):
            np_constrain_entity[idx, 0:len(entity_constrain_list[idx])] = entity_constrain_list[idx]

        all_constraint_prompt = set()
        if "grail" in self.data_path:
            triple_list = []
            entity_dict = {}
            for each_triple in entry['kbs']:
                tmp_subject = entry["kbs"][each_triple][0]
                for each_po in entry['kbs'][each_triple][2]:
                    if each_po[0] not in self.prevent_predicates:
                        if "function : " in each_po[0] and "." not in each_po[0] and "_" not in each_po[0]:
                            function_and_name = each_po[0].lower().split(":")
                            tmp_predicate = function_and_name[1] + " : " + function_and_name[0]
                            if ">=" in each_po[0]:
                                tmp_predicate = tmp_predicate.replace(">=", "greater or equal")
                            elif ">" in each_po[0]:
                                tmp_predicate = tmp_predicate.replace(">", "greater")
                            elif "<=" in each_po[0]:
                                tmp_predicate = tmp_predicate.replace("<=", "less or equal")
                            elif "<" in each_po[0]:
                                tmp_predicate = tmp_predicate.replace("<", "less")
                            elif "arg" in each_po[0]:
                                tmp_predicate = tmp_predicate.replace("argmax", "maximal")
                                tmp_predicate = tmp_predicate.replace("argmin", "minimal")
                        else:
                            tmp_predicate = each_po[0].lower()
                        tmp_predicate = tmp_predicate.replace(" : ", " < name > ")
                        tmp_object = each_po[1]
                        triple_list.append([tmp_subject, tmp_predicate, tmp_object])
                    if each_po[0] in self.property_predicates:
                        tmp_property = each_po[1].lower().replace(" : ", " < class > ")
                        if tmp_subject not in entity_dict:
                            entity_dict[tmp_subject] = tmp_property
                        elif tmp_property not in entity_dict[tmp_subject]:
                            entity_dict[tmp_subject] += tmp_property
            for each_triple in triple_list:
                if each_triple[0] in entity_dict:
                    each_triple[0] = entity_dict[each_triple[0]]
                else:
                    each_triple[0] = each_triple[0].lower().replace(" : ", " < class > ")
                    each_triple[0] = each_triple[0].lower().replace("^^", " < class > ")
                if each_triple[2] in entity_dict:
                    each_triple[2] = entity_dict[each_triple[2]]
                else:
                    each_triple[2] = each_triple[2].lower().replace(" : ", " < class > ")
                    each_triple[2] = each_triple[2].lower().replace("^^", " < class > ")
                all_constraint_prompt.add("predicate: " + each_triple[1] + ", subject: " + each_triple[0] + ", object: "
                                          + each_triple[2] + ", sentence: ")
                # all_constraint_prompt.add(tmp_predicate)
        elif "webnlg17" in self.data_path or "dart" in self.data_path:
            for each_relation in entry["kbs"]:
                subject_name = entry["kbs"][each_relation][0].lower()
                for each_po in entry["kbs"][each_relation][2]:
                    predicate_name = each_po[0]
                    if predicate_name not in self.prevent_predicates:
                        object_name = each_po[1].lower()
                        all_constraint_prompt.add("predicate: " + predicate_name + ", subject: " + subject_name +
                                                  ", object: " + object_name + ", sentence: ")
        elif "spq" in self.data_path:
            # spq only has one triple so take first one
            possible_entity_list = set()
            relation_id = 0
            subject_id = ""
            spq_predicate = ""
            for each_relation in entry['kbs']:
                if relation_id == 0:
                    # because each simple_question only has one predicate useful
                    subject_id = entry["kbs"][each_relation][0]
                    spq_predicate = entry["kbs"][each_relation][2][0][0]
                elif entry["kbs"][each_relation][0] == subject_id:
                    for each_po in entry["kbs"][each_relation][2]:
                        if each_po[0] == "type . object . name" or each_po[0] == "common . topic . alias":
                            possible_entity_list.add(each_po[1])
                relation_id += 1
            possible_entity_list = ", ".join(possible_entity_list)
            all_constraint_prompt.add("predicate: " + spq_predicate +
                                      ", subject: " + possible_entity_list + ", sentence: ")
        constrain_prompt_list = []
        constrain_prompt_len = 0
        if self.data_type != "train":
            for each in all_constraint_prompt:
                constrain_prompt_list.append(each)
                con_len = len(constrain_prompt_list[-1])
                if con_len > constrain_prompt_len:
                    constrain_prompt_len = con_len
        # np_constrain_predicate = np.zeros([len(constrain_prompt_list), constrain_prompt_len])
        # for idx in range(0, len(constrain_prompt_list)):
        #     np_constrain_predicate[idx, 0:len(constrain_prompt_list[idx])] = constrain_prompt_list[idx]
        # np_constrain_predicate = np.array(constrain_prompt_list)
        strings_label = []
        node_ids = []
        edge_ids = []
        strings_label_tokens = ''
        # mark_entity: entities with KB numbers which are important for this task
        # text_entity: entities without KB numbers but only with text, which are less important
        mark_entity = [entry['kbs'][ele_entity][0] for ele_entity in entities]
        mark_entity_number = entities
        text_entity, text_relation = self.get_all_entities_per_sample(mark_entity_number, mark_entity, entry)
        entity_change, relation_change = self.get_change_per_sample(mark_entity, text_entity, text_relation)
        total_entity = mark_entity + text_entity
        adj_matrix = [[-1] * (self.args.max_node_length + 1) for _ in range(self.args.max_node_length + 1)]
        cnt_edge = 0
        if 'title' in entry:
            entity = self.knowledge[entry['title_kb_id']]
            string_label, string_label_tokens, nodes, edges, cnt_edge, adj_matrix = self.linearize_v2(
                entity,
                entity_change,
                self.head_ids,
                self.rel_ids, self.tail_ids,
                relation_change, cnt_edge, adj_matrix)
            strings_label += string_label
            strings_label_tokens += string_label_tokens

        for i, entity_id in enumerate(entities):
            entity = entry['kbs'][entity_id]

            string_label, string_label_tokens, nodes, edges, cnt_edge, adj_matrix = self.linearize_v2(
                entity,
                entity_change,
                self.head_ids,
                self.rel_ids, self.tail_ids,
                relation_change, cnt_edge, adj_matrix)

            strings_label += string_label
            strings_label_tokens += string_label_tokens
            node_ids += nodes
            edge_ids += edges
        words_label_ids, words_label_tokens, words_input_ids, words_input_tokens = [], '', [], ''
        current_text = random.choice(entry['text'])

        for word in current_text.split():
            word_label_ids = self.tokenizer.encode(" {}".format(word), add_special_tokens=False)
            word_label_tokens = copy.deepcopy(word)

            words_label_ids += word_label_ids
            words_label_tokens += ' ' + word_label_tokens

        input_ids_ar, attn_mask_ar, decoder_label_ids, decoder_attn_mask, input_node_ids_ar, input_edge_ids_ar = \
            self.ar_prep_data(words_label_ids, strings_label, self.add_bos_id, self.graph_ids,
                              self.text_ids, node_ids, edge_ids)

        node_length_ar = max(input_node_ids_ar) + 1
        edge_length_ar = max(input_edge_ids_ar) + 1

        def masked_fill(src, masked_value, fill_value):
            return [src[src_id] if src[src_id] != masked_value and src[src_id] < fill_value else fill_value for src_id
                    in range(len(src))]

        input_node_ids_ar, input_edge_ids_ar = masked_fill(input_node_ids_ar, -1, self.args.max_node_length), \
                                               masked_fill(input_edge_ids_ar, -1, self.args.max_edge_length)

        def masked_fill_matrix(adj_matrix_input, masked_value, fill_value):
            adj_matrix_tmp = copy.deepcopy(adj_matrix_input)
            for a_id in range(len(adj_matrix_tmp)):
                for b_id in range(len(adj_matrix_tmp)):
                    if adj_matrix_tmp[a_id][b_id] == masked_value or adj_matrix_tmp[a_id][b_id] > fill_value:
                        adj_matrix_tmp[a_id][b_id] = fill_value
            return adj_matrix_tmp

        adj_matrix_ar = masked_fill_matrix(adj_matrix, -1, self.args.max_edge_length)
        assert len(input_ids_ar) == len(attn_mask_ar) == self.args.max_input_length == len(input_node_ids_ar) == len(
            input_edge_ids_ar)
        assert len(decoder_label_ids) == len(decoder_attn_mask) == self.args.max_output_length
        input_ids_ar = torch.LongTensor(input_ids_ar)
        attn_mask_ar = torch.LongTensor(attn_mask_ar)
        decoder_label_ids = torch.LongTensor(decoder_label_ids)
        decoder_attn_mask = torch.LongTensor(decoder_attn_mask)
        input_node_ids_ar = torch.LongTensor(input_node_ids_ar)
        input_edge_ids_ar = torch.LongTensor(input_edge_ids_ar)
        node_length_ar = torch.LongTensor([node_length_ar])
        edge_length_ar = torch.LongTensor([edge_length_ar])
        adj_matrix_ar = torch.LongTensor(adj_matrix_ar)
        entity_constrain_list = torch.LongTensor(np_constrain_entity)
        # constrain_prompt_list = torch.Tensor(np_constrain_predicate)
        return input_ids_ar, attn_mask_ar, decoder_label_ids, decoder_attn_mask, \
               input_node_ids_ar, input_edge_ids_ar, node_length_ar, edge_length_ar, adj_matrix_ar, \
               entity_constrain_list, constrain_prompt_list


def evaluate_bleu(data_ref, data_sys):
    coco_eval = run_coco_eval(data_ref, data_sys)
    scores = {metric: score for metric, score in list(coco_eval.eval.items())}
    # return scores["Bleu_4"]
    return scores

    # (Bleu(), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
    # (Meteor(), "METEOR"),
    # (Rouge(), "ROUGE_L"),
    # (Cider(), "CIDEr")
