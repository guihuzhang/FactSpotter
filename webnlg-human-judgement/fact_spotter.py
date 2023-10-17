import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
from datasets import Dataset
import json
from os.path import exists
import sys


def read_triples(filepath):
    rb = open(filepath, "r")
    triples = {}
    count = 0
    for line in rb.readlines():
        count += 1
        triples[count] = []

        for triple in line.split("<br>"):

            if len(triple) == 0:
                continue
            triple = triple.split("|")
            subject = " ".join(triple[0].split("_"))
            object = " ".join(triple[2].split("_"))
            predicate = ""
            last = 0
            for i in range(len(triple[1])):
                c = triple[1][i]
                if c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                    predicate = predicate + " " + triple[1][last:i]
                    last = i
            predicate = predicate + " " + triple[1][last:-1]
            triples[count].append(
                "predicate: " + predicate.lower().strip() + ", subject: " + subject.strip() +
                ", object: " + object.strip())

    return triples


def test_submission(filepath, ground_truth):
    rb = open(filepath, "r", encoding="utf-8")
    input_sentences = []
    count = 1
    ids = []
    for line in rb.readlines():
        sentence = line.strip()
        if len(ground_truth) < count:
            break
        for triple in ground_truth[count]:
            input_sentences.append({"text": triple + ", sentence: " + sentence})
            ids.append(count)
        count += 1
    return input_sentences, ids


if __name__ == '__main__':

    # Load Tokenizer and Model
    parameters = json.load(open(sys.argv[1]))
    tokenizer = AutoTokenizer.from_pretrained(parameters["model-base"])
    model = AutoModelForSequenceClassification.from_pretrained(parameters["model-base"])

    model_state_dict = torch.load(parameters["model-finetuned-path"])
    model.load_state_dict(model_state_dict)

    ground_truth = read_triples(parameters["triples"])

    files = [f.path for f in os.scandir(parameters["teams-samples"])]
    for file_name in files:
        print(file_name)
        test, ids = test_submission(file_name, ground_truth)
        test_dataset = Dataset.from_list(test)
        test_dataset = test_dataset.map(
            lambda example: tokenizer(example["text"], return_tensors="pt", truncation=True, padding='max_length',
                                      max_length=512), batched=False)
        total = []
        count = 0
        print(len(test))
        print(len(ids))
        with torch.no_grad():
            for item in test_dataset:
                input_ids = torch.Tensor(item["input_ids"]).to(torch.int64)
                att_ids = torch.Tensor(item["attention_mask"]).to(torch.int64)
                tmp_pred = model(input_ids=input_ids, attention_mask=att_ids)
                _, pred_index = tmp_pred.logits.max(dim=1, )
                softmax_cls_output = torch.softmax(tmp_pred.logits, dim=1, )
                total.append(softmax_cls_output[:, 1].item())
        sum_total = 0.0
        sum_partial = 0.0
        count_partial = 0
        count_total = 0
        last_index = ids[0]
        scores = []
        for i in range(len(ids)):
            index = ids[i]
            if last_index != index:
                sum_total += (sum_partial / count_partial)
                scores.append(sum_partial / count_partial)
                sum_partial = total[i]
                count_partial = 1
                count_total += 1
            else:
                sum_partial += total[i]
                count_partial += 1
            last_index = index
        sum_total += (sum_partial / count_partial)
        scores.append(sum_partial / count_partial)
        folder_path = parameters["metrics-path"] + "/fact-spotter"
        if not exists(folder_path):
            os.mkdir(folder_path)
        wb = open(folder_path + "/" + file_name.split('/')[-1], "w")
        for s in scores:
            wb.write(str(s) + "\n")
        wb.close()
        count_total += 1
        print(count_total)
        print("Per sentence score: " + str(sum_total / count_total))
        print("Per triple score: " + str(sum(total) / len(total)))
