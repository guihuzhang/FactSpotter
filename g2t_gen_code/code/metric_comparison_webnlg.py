import csv
import json
import sys
import gc
import tensorflow as tf
import numpy as np
import torch
from bert_score import score as score_bert
from bleurt import score as score_bleurt
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from eval_code.bart_score import BARTScorer
from eval_code.pycocoevalcap.eval import COCOEvalCap
from eval_code.pycocotools.coco import COCO


def create_coco_refs(coco_ref):
    """Create MS-COCO human references JSON."""
    out = {'info': {}, 'licenses': [], 'images': [], 'type': 'captions', 'annotations': []}
    ref_id = 0
    for inst_id, refs in enumerate(coco_ref):
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


def run_coco_eval(coco_ref, data_sys):
    """Run the COCO evaluator, return the resulting evaluation object (contains both
    system- and segment-level scores."""
    # convert references and system outputs to MS-COCO format in-memory
    coco_ref = create_coco_refs(coco_ref)
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
    cls_tokenizer = AutoTokenizer.from_pretrained("google/electra-small-discriminator")
    cls_model = AutoModelForSequenceClassification.from_pretrained("google/electra-small-discriminator")
    # cls model directory
    cls_state_dict = torch.load("electra_webnlg_fix.5pred.25ent.25both.1drop0.9888319088319089.pt")
    cls_model.load_state_dict(cls_state_dict)
    cls_model = cls_model.to(torch.device("cuda"))
    cls_model.eval()

    prevent_predicates = ["class name", "entity name", "domains", "literal name", "sentence category",
                          "Sentence Data Source", ]
    property_predicates = ["class name", "entity name", "literal name"]

    # input1: baseline result by lines
    cls_positive_threshold = .5
    baseline_res = read_gen_result(
        "webnlg_const_t5_jointgt_full/webnlg_baseline_joint_predictions.txt")
    # input2: generation with constraints by lines
    constrained_res = read_gen_result(
        "webnlg_const_t5_jointgt_full/webnlg_const5beam.15weight_predictions.txt")
    output_list = []
    # dataset file
    with open("webnlg_const_kun/test.json", 'r') as load_f:
        gt_list = json.load(load_f)
        data_ref = [[x.strip(" ?.").lower() for x in data_ele['text']]
                    for data_ele in gt_list]
        gt_cls_input = []
        baseline_cls_input = []
        constrained_cls_input = []
        gt_cls_id_list = []
        cls_id_list = []
        input_id = 0
        all_triples = []
        # baseline_res = [baseline_res[x["id"]] for x in gt_list]
        # constrained_res = [constrained_res[x["id"]] for x in gt_list]
        print("Calculating Classification Score!")
        for idx in range(0, len(gt_list)):
            all_constraint_prompt = set()
            for each_triple in gt_list[idx]['kbs']:
                tmp_subject = gt_list[idx]["kbs"][each_triple][0].lower()
                for each_po in gt_list[idx]['kbs'][each_triple][2]:
                    tmp_predicate = each_po[0]
                    if each_po[0] not in prevent_predicates:
                        tmp_object = each_po[1].lower()
                        all_constraint_prompt.add("predicate: " + tmp_predicate + ", subject: " + tmp_subject +
                                                  ", object: " + tmp_object + ", sentence: ")
            all_triples.append(all_constraint_prompt)
            # gt = gt_list[idx]["text"][0]
            for each_prompt in all_constraint_prompt:
                for each_gt in gt_list[idx]["text"]:
                    gt_cls_input.append(each_prompt + each_gt.lower().strip(",? "))
                    gt_cls_id_list.append(input_id)
                tmp_base = baseline_res[idx]
                baseline_cls_input.append(each_prompt + tmp_base.strip(",? "))
                tmp_con = constrained_res[idx]
                constrained_cls_input.append(each_prompt + tmp_con.strip(",? "))
                cls_id_list.append(input_id)
            input_id += 1
        # Kun: optimise to be batch eval to make it faster
        torch.cuda.empty_cache()
        tmp_batch = 16
        # get cls score for each batch of GT
        batched_gt_cls = [gt_cls_input[i:i + tmp_batch] for i in range(0, len(gt_cls_input), tmp_batch)]
        cls_gt = []
        cls_gt_dict = {}
        cls_gt_neg = {}
        for golden_batch in tqdm(batched_gt_cls, 'GT CLS Progress'):
            tmp_cls = sentence_cls_score(golden_batch, cls_model, cls_tokenizer)
            cls_gt.extend([float(x) for x in tmp_cls])
        # allocate cls results to each sentence id
        for idx in range(0, len(gt_cls_id_list)):
            tmp_id = gt_cls_id_list[idx]
            tmp_score = cls_gt[idx]
            if tmp_score < .5:
                if tmp_id not in cls_gt_neg:
                    cls_gt_neg[tmp_id] = [gt_cls_input[idx]]
                else:
                    cls_gt_neg[tmp_id].append(gt_cls_input[idx])
            if tmp_id not in cls_gt_dict:
                cls_gt_dict[tmp_id] = [tmp_score]
            else:
                cls_gt_dict[tmp_id].append(tmp_score)
        for each_id in cls_gt_dict:
            cls_gt_dict[each_id] = np.mean(cls_gt_dict[each_id])
        # get cls score for each batch of baseline
        batched_baseline_cls = [baseline_cls_input[i:i + tmp_batch] for i in
                                range(0, len(baseline_cls_input), tmp_batch)]
        cls_baseline = []
        cls_baseline_binary = []
        cls_baseline_dict = {}
        cls_baseline_neg = {}
        for golden_batch in tqdm(batched_baseline_cls, 'Baseline CLS Progress'):
            tmp_cls = sentence_cls_score(golden_batch, cls_model, cls_tokenizer)
            cls_baseline.extend([float(x) for x in tmp_cls])
            cls_baseline_binary.extend([1 if x >= .5 else 0 for x in tmp_cls])
        # allocate cls results to each sentence id
        for idx in range(0, len(cls_id_list)):
            tmp_id = cls_id_list[idx]
            tmp_score = cls_baseline[idx]
            if tmp_score < .5:
                if tmp_id not in cls_baseline_neg:
                    cls_baseline_neg[tmp_id] = [baseline_cls_input[idx]]
                else:
                    cls_baseline_neg[tmp_id].append(baseline_cls_input[idx])
            if tmp_id not in cls_baseline_dict:
                cls_baseline_dict[tmp_id] = [tmp_score]
            else:
                cls_baseline_dict[tmp_id].append(tmp_score)
        cls_baseline_by_num = {}
        for each_id in cls_baseline_dict:
            tmp_len = len(cls_baseline_dict[each_id])
            if tmp_len not in cls_baseline_by_num:
                cls_baseline_by_num[tmp_len] = cls_baseline_dict[each_id]
            else:
                cls_baseline_by_num[tmp_len].extend(cls_baseline_dict[each_id])
            cls_baseline_dict[each_id] = np.mean(cls_baseline_dict[each_id])
        for each_num in cls_baseline_by_num:
            cls_baseline_by_num[each_num] = np.mean(cls_baseline_by_num[each_num])
        batched_constraint_cls = [constrained_cls_input[i:i + tmp_batch]
                                  for i in range(0, len(constrained_cls_input), tmp_batch)]
        cls_constraint = []
        cls_constraint_binary = []
        cls_constraint_dict = {}
        cls_cons_neg = {}
        for golden_batch in tqdm(batched_constraint_cls, 'Constraint CLS Progress'):
            tmp_cls = sentence_cls_score(golden_batch, cls_model, cls_tokenizer)
            cls_constraint.extend([float(x) for x in tmp_cls])
            cls_constraint_binary.extend([1 if x >= .5 else 0 for x in tmp_cls])
        # allocate cls results to each sentence id
        for idx in range(0, len(cls_id_list)):
            tmp_id = cls_id_list[idx]
            tmp_score = cls_constraint[idx]
            if tmp_score < .5:
                if tmp_id not in cls_cons_neg:
                    cls_cons_neg[tmp_id] = [constrained_cls_input[idx]]
                else:
                    cls_cons_neg[tmp_id].append(constrained_cls_input[idx])
            if tmp_id not in cls_constraint_dict:
                cls_constraint_dict[tmp_id] = [tmp_score]
            else:
                cls_constraint_dict[tmp_id].append(tmp_score)
        cls_constraint_by_num = {}
        for each_id in cls_constraint_dict:
            tmp_len = len(cls_constraint_dict[each_id])
            if tmp_len not in cls_constraint_by_num:
                cls_constraint_by_num[tmp_len] = cls_constraint_dict[each_id]
            else:
                cls_constraint_by_num[tmp_len].extend(cls_constraint_dict[each_id])
            cls_constraint_dict[each_id] = np.mean(cls_constraint_dict[each_id])
        for each_num in cls_constraint_by_num:
            cls_constraint_by_num[each_num] = np.mean(cls_constraint_by_num[each_num])
        tmp_ref = [x[0] for x in data_ref]
        # calc bert score
        torch.cuda.empty_cache()
        bert_p_baseline, bert_r_baseline, bert_f1_baseline = score_bert(
            baseline_res, data_ref, model_type="roberta-large", verbose=True)
        bert_p_constrained, bert_r_constrained, bert_f1_constrained = score_bert(
            constrained_res, data_ref, model_type="roberta-large", verbose=True)

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

        bleurt_ref_id = []
        bleurt_refs = []
        bleurt_base_res = []
        bleurt_con_res = []
        for ref_idx in range(0, len(data_ref)):
            for each_ref in data_ref[ref_idx]:
                bleurt_ref_id.append(ref_idx)
                bleurt_refs.append(each_ref)
                bleurt_base_res.append(baseline_res[ref_idx])
                bleurt_con_res.append(constrained_res[ref_idx])
        bleurt_scorer = score_bleurt.LengthBatchingBleurtScorer("BLEURT-20-D12")
        bleurt_baseline = bleurt_scorer.score(references=bleurt_refs, candidates=bleurt_base_res, batch_size=8)
        bleurt_constraint = bleurt_scorer.score(references=bleurt_refs, candidates=bleurt_con_res, batch_size=8)

        bleurt_baseline_dict = {}
        # allocate bleurt results to each sentence id
        for idx in range(0, len(bleurt_ref_id)):
            tmp_id = bleurt_ref_id[idx]
            tmp_score = bleurt_baseline[idx]
            if tmp_id not in bleurt_baseline_dict:
                bleurt_baseline_dict[tmp_id] = [tmp_score]
            else:
                bleurt_baseline_dict[tmp_id].append(tmp_score)
        for each_id in bleurt_baseline_dict:
            bleurt_baseline_dict[each_id] = np.max(bleurt_baseline_dict[each_id])

        bleurt_cons_dict = {}
        # allocate bleurt results to each sentence id
        for idx in range(0, len(bleurt_ref_id)):
            tmp_id = bleurt_ref_id[idx]
            tmp_score = bleurt_constraint[idx]
            if tmp_id not in bleurt_cons_dict:
                bleurt_cons_dict[tmp_id] = [tmp_score]
            else:
                bleurt_cons_dict[tmp_id].append(tmp_score)
        for each_id in bleurt_cons_dict:
            bleurt_cons_dict[each_id] = np.max(bleurt_cons_dict[each_id])

        tf.keras.backend.clear_session()
        gc.collect()

        # Kun: get traditional metrics also by batch! deprecate old one-by-one silly code!
        coco_eval_base = run_coco_eval(data_ref, baseline_res)
        coco_eval_con = run_coco_eval(data_ref, constrained_res)
        coco_detail_baseline = coco_eval_base.imgToEval
        coco_detail_constraint = coco_eval_con.imgToEval
        for idx in range(0, len(gt_list)):
            relation = gt_list[idx]["kbs"]
            triples = all_triples[idx]
            gt = gt_list[idx]["text"][0]
            tmp_base = baseline_res[idx]
            tmp_con = constrained_res[idx]

            tmp_cls_gt = cls_gt_dict[idx]
            tmp_cls_base = cls_baseline_dict[idx]
            tmp_cls_con = cls_constraint_dict[idx]

            gt_neg = cls_gt_neg[idx] if idx in cls_gt_neg else ""
            cons_neg = cls_cons_neg[idx] if idx in cls_cons_neg else ""
            baseline_neg = cls_baseline_neg[idx] if idx in cls_baseline_neg else ""

            tmp_bleurt_base = bleurt_baseline_dict[idx]
            tmp_bleurt_constraint = bleurt_cons_dict[idx]

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

            output_list.append([triples, gt, tmp_cls_gt, gt_neg,
                                tmp_base, tmp_cls_base, baseline_neg, tmp_bleurt_base, tmp_bart_base, tmp_bert_base,
                                tmp_bleu4_base, tmp_meteor_base, coco_metrics_base,
                                tmp_con, tmp_cls_con, cons_neg, tmp_bleurt_constraint, tmp_bart_constraint,
                                tmp_bert_con, tmp_bleu4_con, tmp_meteor_con, coco_metrics_con,
                                tmp_cls_con - tmp_cls_base, tmp_cls_gt - tmp_cls_base, tmp_cls_gt - tmp_cls_con,
                                tmp_bleurt_constraint - tmp_bleurt_base, tmp_bart_constraint - tmp_bart_base,
                                tmp_bert_con - tmp_bert_base, tmp_bleu4_con - tmp_bleu4_base,
                                tmp_meteor_con - tmp_meteor_base])
        print("GT CLS", np.mean(cls_gt))
        print("positive GT CLS:", np.mean([1 if cls_gt_dict[x] > cls_positive_threshold else 0
                                           for x in cls_gt_dict]))
        print("Sentences with neg in GT", len(cls_gt_neg)/len(gt_list))
        print("Baseline CLS", np.mean(cls_baseline), "Baseline CLS Bin", np.mean(cls_baseline_binary),
              "BERT", np.mean(bert_r_baseline), "BART", np.mean(bart_baseline),
              "BLEURT", np.mean(bleurt_baseline), coco_eval_base.eval)
        print(cls_baseline_by_num)
        print("positive Baseline CLS:", np.mean([1 if cls_baseline_dict[x] > cls_positive_threshold else 0
                                                 for x in cls_baseline_dict]))
        print("Sentences with neg in Baseline", len(cls_baseline_neg)/len(baseline_res))

        print("Constraint CLS", np.mean(cls_constraint), "Constraint CLS Bin", np.mean(cls_constraint_binary),
              "BERT", np.mean(bert_r_constrained), "BART", np.mean(bart_constraint),
              "BLEURT", np.mean(bleurt_constraint), coco_eval_con.eval)
        print(cls_constraint_by_num)
        print("positive Constraint CLS:", np.mean([1 if cls_constraint_dict[x] > cls_positive_threshold else 0
                                                   for x in cls_constraint_dict]))
        print("Sentences with neg in Cons", len(cls_cons_neg)/len(constrained_res))

    # output file name
    with open('WEBNLGRedo.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        # Title line of table
        writer.writerow(["Predicate", "GT", "GT Pred CLS", "GT Neg",
                         "Baseline", "Base Pred CLS", "Base Neg", "Base BLEURT", "Base BART", "Base BERT", "Base BLEU4",
                         "Base Meteor", "Base Other COCO",
                         "After Constrain", "Cons Pred CLS", "Cons Neg", "Cons BLEURT", "Cons BART", "Cons BERT",
                         "Cons BLEU4", "Cons Meteor", "Cons Other COCO",
                         "Cons-Baseline CLS Gap", "GT-Baseline CLS Gap", "GT-Cons CLS Gap",
                         "Cons-Base BLEURT gap", "Cons-Base BART gap",
                         "Cons-Baseline BERT Gap", "Cons-Baseline BLEU Gap",
                         "Cons-Baseline METEOR Gap"])
        writer.writerows(output_list)
