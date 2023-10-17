import csv
import json
import sys

import gc
import tensorflow as tf
import numpy as np
import torch
from bleurt import score as score_bleurt
from bert_score import score
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from eval_code.bart_score import BARTScorer

from eval_code.pycocoevalcap.eval import COCOEvalCap
from eval_code.pycocotools.coco import COCO


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


def read_gen_result(generated_file):
    with open(generated_file, 'rt') as f:
        return [x.replace("\n", "").lower() for x in f]


def sentence_cls_score(input_string, predicate_cls_model, predicate_cls_tokenizer):
    tokenized_cls_input = predicate_cls_tokenizer(input_string, truncation=True)
    tokenized_cls_input["input_ids"] = pad_sequence(
        [torch.LongTensor(x) for x in tokenized_cls_input["input_ids"]],
        padding_value=predicate_cls_tokenizer.pad_token_id, batch_first=True).to(torch.device("cuda"))
    tokenized_cls_input["attention_mask"] = pad_sequence(
        [torch.LongTensor(x) for x in tokenized_cls_input["attention_mask"]], padding_value=0,
        batch_first=True).to(torch.device("cuda"))
    prev_cls_output = predicate_cls_model(input_ids=tokenized_cls_input["input_ids"],
                                          attention_mask=tokenized_cls_input["attention_mask"])
    softmax_cls_output = torch.softmax(prev_cls_output.logits, dim=1, )
    return softmax_cls_output[:, 1]


if __name__ == '__main__':
    # previous grail cls model, to be updated to new version
    cls_tokenizer = AutoTokenizer.from_pretrained("google/electra-small-discriminator")
    cls_model = AutoModelForSequenceClassification.from_pretrained("google/electra-small-discriminator")
    # cls model directory
    cls_state_dict = torch.load("electra_spq.5pred.25ent.25both.1drop0.9551447538262954.pt")
    cls_model.load_state_dict(cls_state_dict)
    cls_model = cls_model.to(torch.device("cuda"))
    cls_model.eval()
    cls_positive_threshold = .5
    # input1: baseline result by lines
    # I overwrote zero-shot baseline file! This needs to be updated
    baseline_res = read_gen_result(
        "spq_predicate_zero_t5_jointgt/spq_predicate_zero_baseline_new_predictions.txt")
    # input2: generation with constraints by lines
    constrained_res = read_gen_result(
        "spq_predicate_zero_t5_jointgt/spq_predicate_zero_baseline_t5jointgt.15_predictions.txt")
    output_list = []
    # dataset file
    with open("spq_predicate_zero/test.json", 'r') as load_f:
        gt_list = json.load(load_f)
        data_ref = [[x.split("?")[0].strip(" ?.").lower() for x in data_ele['text']]
                    for data_ele in gt_list]
        print("Calculating Classification Score!")
        gt_cls_input = []
        baseline_cls_input = []
        constrained_cls_input = []
        all_triples = []
        for idx in range(0, len(gt_list)):
            all_constraint_prompt = set()
            relation = gt_list[idx]["kbs"]
            possible_entity_list = set()
            relation_id = 0
            subject_id = ""
            spq_predicate = ""
            for each_relation in relation:
                if relation_id == 0:
                    subject_id = relation[each_relation][0]
                    spq_predicate = relation[each_relation][2][0][0]
                elif relation[each_relation][0] == subject_id:
                    for each_po in relation[each_relation][2]:
                        if each_po[0] == "type . object . name" or each_po[0] == "common . topic . alias":
                            possible_entity_list.add(each_po[1])
                relation_id += 1
            possible_entity_list = ", ".join(possible_entity_list)
            all_constraint_prompt = "predicate: " + spq_predicate + ", subject: " + \
                                    possible_entity_list + ", sentence: "
            all_triples.append(all_constraint_prompt)
            gt = gt_list[idx]["text"][0]
            gt_cls_input.append(all_constraint_prompt + gt.lower().strip(",? "))
            tmp_base = baseline_res[idx]
            baseline_cls_input.append(all_constraint_prompt + tmp_base.strip(",? "))
            tmp_con = constrained_res[idx]
            constrained_cls_input.append(all_constraint_prompt + tmp_con.strip(",? "))
        # Kun: optimise to be batch eval to make it faster
        torch.cuda.empty_cache()
        tmp_batch = 16
        # get cls score for each batch of GT
        batched_gt_cls = [gt_cls_input[i:i + tmp_batch] for i in range(0, len(gt_cls_input), tmp_batch)]
        cls_gt = []
        for golden_batch in tqdm(batched_gt_cls, 'GT CLS Progress'):
            tmp_depth = sentence_cls_score(golden_batch, cls_model, cls_tokenizer)
            cls_gt.extend([float(x) for x in tmp_depth])
        # get cls score for each batch of baseline
        batched_baseline_cls = [baseline_cls_input[i:i + tmp_batch] for i in range(0, len(baseline_cls_input), tmp_batch)]

        cls_baseline = []
        for golden_batch in tqdm(batched_baseline_cls, 'Baseline CLS Progress'):
            tmp_depth = sentence_cls_score(golden_batch, cls_model, cls_tokenizer)
            cls_baseline.extend([float(x) for x in tmp_depth])
        batched_constraint_cls = [constrained_cls_input[i:i + tmp_batch]
                                  for i in range(0, len(constrained_cls_input), tmp_batch)]

        cls_constraint = []
        for golden_batch in tqdm(batched_constraint_cls, 'Constraint CLS Progress'):
            tmp_depth = sentence_cls_score(golden_batch, cls_model, cls_tokenizer)
            cls_constraint.extend([float(x) for x in tmp_depth])
        # calc bert score

        torch.cuda.empty_cache()
        tmp_ref = [x[0] for x in data_ref]
        bert_p_baseline, bert_r_baseline, bert_f1_baseline = score(
            baseline_res, tmp_ref, model_type="roberta-large", verbose=True)
        bert_p_constrained, bert_r_constrained, bert_f1_constrained = score(
            constrained_res, tmp_ref, model_type="roberta-large", verbose=True)

        bert_r_baseline = bert_r_baseline.numpy()
        bert_r_constrained = bert_r_constrained.numpy()

        torch.cuda.empty_cache()
        bart_scorer = BARTScorer()
        bart_scorer.load("bart_score.pth")
        # bart_scorer = BARTScorer(device='cuda:0', checkpoint='facebook/bart-large-cnn')
        # List[List of references for each test sample]
        bart_baseline = bart_scorer.multi_ref_score(baseline_res, data_ref, agg="max", batch_size=tmp_batch)
        bart_constraint = bart_scorer.multi_ref_score(constrained_res, data_ref, agg="max", batch_size=tmp_batch)
        torch.cuda.empty_cache()

        bleurt_scorer = score_bleurt.BleurtScorer("BLEURT-20-D12")
        bleurt_baseline = bleurt_scorer.score(references=tmp_ref, candidates=baseline_res, batch_size=1)
        bleurt_constraint = bleurt_scorer.score(references=tmp_ref, candidates=constrained_res, batch_size=1)
        tf.keras.backend.clear_session()
        gc.collect()
        # Kun: get traditional metrics also by batch! deprecate old one-by-one silly code!
        coco_eval_base = run_coco_eval(data_ref, baseline_res)
        coco_eval_con = run_coco_eval(data_ref, constrained_res)
        coco_detail_baseline = coco_eval_base.imgToEval
        coco_detail_constraint = coco_eval_con.imgToEval
        for idx in range(0, len(gt_list)):
            relation = gt_list[idx]["kbs"]
            predicate = all_triples[idx]
            gt = gt_list[idx]["text"][0]
            tmp_base = baseline_res[idx]
            tmp_con = constrained_res[idx]

            tmp_cls_gt = cls_gt[idx]
            tmp_cls_base = cls_baseline[idx]
            tmp_cls_con = cls_constraint[idx]

            tmp_bleurt_base = bleurt_baseline[idx]
            tmp_bleurt_constraint = bleurt_constraint[idx]

            tmp_bart_base = bart_baseline[idx]
            tmp_bart_constraint = bart_constraint[idx]

            tmp_bert_base = bert_r_baseline[idx]
            tmp_bert_con = bert_r_constrained[idx]

            coco_metrics_base = coco_detail_baseline["inst-" + str(idx)]
            coco_metrics_con = coco_detail_constraint["inst-" + str(idx)]
            # If we want to cmp some other metrics of COCO, just change following key names to get their value,
            # key supported: Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4', 'METEOR', 'ROUGE_L', 'CIDEr'
            tmp_bleu4_base = coco_metrics_base["Bleu_4"]
            tmp_bleu4_con = coco_metrics_con["Bleu_4"]

            tmp_meteor_base = coco_metrics_base["METEOR"]
            tmp_meteor_con = coco_metrics_con["METEOR"]

            output_list.append([predicate,  gt, tmp_cls_gt,
                                tmp_base, tmp_cls_base, tmp_bleurt_base, tmp_bart_base, tmp_bert_base, tmp_bleu4_base,
                                tmp_meteor_base, coco_metrics_base,
                                tmp_con, tmp_cls_con, tmp_bleurt_constraint, tmp_bart_constraint, tmp_bert_con,
                                tmp_bleu4_con, tmp_meteor_con, coco_metrics_con,
                                tmp_cls_con - tmp_cls_base, tmp_cls_gt - tmp_cls_base, tmp_cls_gt - tmp_cls_con,
                                tmp_bleurt_constraint - tmp_bleurt_base, tmp_bart_constraint - tmp_bart_base,
                                tmp_bert_con - tmp_bert_base, tmp_bleu4_con - tmp_bleu4_base,
                                tmp_meteor_con - tmp_meteor_base])
        print("GT CLS", np.mean(cls_gt))
        print("Baseline CLS", np.mean(cls_baseline), "BERT", np.mean(bert_r_baseline), "BART", np.mean(bart_baseline),
              "BLEURT", np.mean(bleurt_baseline), coco_eval_base.eval)
        print("positive Baseline CLS:", np.mean([1 if x > cls_positive_threshold else 0 for x in cls_baseline]))
        print("Constraint CLS", np.mean(cls_constraint), "BERT", np.mean(bert_r_constrained), "BART",
              np.mean(bart_constraint), "BLEURT", np.mean(bleurt_constraint), coco_eval_con.eval)
        print("positive Constraint CLS:", np.mean([1 if x > cls_positive_threshold else 0 for x in cls_constraint]))

    # output file name
    with open('SPQUpdated.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        # Title line of table
        writer.writerow(["Predicate", "GT", "GT Pred CLS",
                         "Baseline", "Base Pred CLS", "Base BLEURT", "Base BART", "Base BERT", "Base BLEU4",
                         "Base Meteor", "Base Other COCO",
                         "After Constrain", "Cons Pred CLS", "Cons BLEURT", "Cons BART", "Cons BERT", "Cons BLEU4",
                         "Cons Meteor", "Cons Other COCO",
                         "Cons-Baseline CLS Gap", "GT-Baseline CLS Gap", "GT-Cons CLS Gap",
                         "Cons-Base BLEURT gap", "Cons-Base BART gap",
                         "Cons-Baseline BERT Gap", "Cons-Baseline BLEU Gap",
                         "Cons-Baseline METEOR Gap"])
        writer.writerows(output_list)


