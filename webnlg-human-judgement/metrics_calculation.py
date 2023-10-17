import sys
import codecs
import copy
import logging
import nltk

import os
from os.path import exists
import xml.etree.ElementTree as ET
import json
from bert_score import score
from metrics.bleurt.bleurt import score as bleurt_score

from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction, sentence_bleu
from nltk.translate.meteor_score import meteor_score
from razdel import tokenize
from metrics.bart_score import BARTScorer

from torchmetrics.text.infolm import InfoLM

BLEU_PATH = 'metrics/multi-bleu-detok.perl'
METEOR_PATH = 'metrics/meteor-1.5/meteor-1.5.jar'


def parse(refs_path, hyps_path, num_refs, lng='en'):
    logging.info('STARTING TO PARSE INPUTS...')
    print('STARTING TO PARSE INPUTS...')
    # references
    references = []
    for i in range(num_refs):
        fname = refs_path + str(i) if num_refs > 1 else refs_path
        with codecs.open(fname, 'r', 'utf-8') as f:
            texts = f.readlines()
            for j, text in enumerate(texts):
                if len(references) <= j:
                    references.append([text])
                else:
                    references[j].append(text)

    # references tokenized
    references_tok = copy.copy(references)
    for i, refs in enumerate(references_tok):
        references_tok[i] = [' '.join(nltk.word_tokenize(ref)) for ref in refs]

    # hypothesis
    with codecs.open(hyps_path, 'r', 'utf-8') as f:
        hypothesis = f.read().split('\n')

    # hypothesis tokenized
    hypothesis_tok = copy.copy(hypothesis)
    hypothesis_tok = [' '.join(nltk.word_tokenize(hyp)) for hyp in hypothesis_tok]

    logging.info('FINISHING TO PARSE INPUTS...')
    print('FINISHING TO PARSE INPUTS...')
    return references, references_tok, hypothesis, hypothesis_tok


def parse_new(references, hypothesis, lng='en'):
    logging.info('STARTING TO PARSE INPUTS...')
    print('STARTING TO PARSE INPUTS...')

    # references tokenized
    references_tok = copy.copy(references)
    for i, refs in enumerate(references_tok):
        if lng == 'ru':
            references_tok[i] = [' '.join([_.text for _ in tokenize(ref)]) for ref in refs]
        else:
            references_tok[i] = [' '.join(nltk.word_tokenize(ref)) for ref in refs]

    # hypothesis tokenized
    hypothesis_tok = copy.copy(hypothesis)
    if lng == 'ru':
        hypothesis_tok = [' '.join([_.text for _ in tokenize(hyp)]) for hyp in hypothesis_tok]
    else:
        hypothesis_tok = [' '.join(nltk.word_tokenize(hyp)) for hyp in hypothesis_tok]

    logging.info('FINISHING TO PARSE INPUTS...')
    print('FINISHING TO PARSE INPUTS...')
    return references, references_tok, hypothesis, hypothesis_tok


def bleu_nltk(references, hypothesis):
    # check for empty lists
    references_, hypothesis_ = [], []
    print(len(references))
    print(len(hypothesis))
    for i, refs in enumerate(references):
        refs_ = [ref for ref in refs if ref.strip() != '']
        if len(refs_) > 0:
            references_.append([ref.split() for ref in refs_])
            hypothesis_.append(hypothesis[i].split())

    chencherry = SmoothingFunction()
    return corpus_bleu(references_, hypothesis_, smoothing_function=chencherry.method3)


def sentence_bleu_nltk(references, hypothesis):
    # check for empty lists
    references_, hypothesis_ = [], []
    for i, refs in enumerate(references):
        refs_ = [ref for ref in refs if ref.strip() != '']
        if len(refs_) > 0:
            references_.append([ref.split() for ref in refs_])
            hypothesis_.append(hypothesis[i].split())

    bleu = []
    print(len(references_))
    print(len(hypothesis_))
    chencherry = SmoothingFunction()
    for i in range(len(references)):
        bleu.append(sentence_bleu(references_[i], hypothesis_[i], smoothing_function=chencherry.method3))
    return bleu


def infolm_score(references, hypothesis):
    infolm = InfoLM('google/bert_uncased_L-2_H-128_A-2', information_measure="fisher_rao_distance", batch_size=16,
                    return_sentence_level_score=True, idf=False)

    results = []
    all_hyp = []
    all_ref = []
    for i in range(len(hypothesis)):
        hyp = [hypothesis[i] for j in range(len(references[i]))]
        all_hyp += hyp
        all_ref += references[i]
        """
		pred = metric_call.evaluate_batch(hyp, references[i])
		pred = min(pred['depth_score'])
		print(pred)
		results.append(pred)
		"""
    pred = infolm(all_hyp, all_ref)
    pred = pred[1]
    count = 0
    for i in range(len(hypothesis)):
        min = 100.0
        for j in range(len(references[i])):
            poz = j + count
            if pred[poz].item() < min:
                min = pred[poz].item()
        results.append(min)
        count += len(references[i])
    return results


def bart_score(references, hypothesis, num_refs):
    bart_scorer = BARTScorer(device='cuda:0', checkpoint='facebook/bart-large-cnn')
    bart_scorer.load(path='bart_score.pth')
    # srcs = ["I'm super happy today.", "This is a good idea."]
    # tgts = [["I feel good today.", "I feel sad today."],
    #             ["Not bad.", "Sounds like a good idea."]]  # List[List of references for each test sample]
    new_references = []
    for i in range(len(hypothesis)):
        refs = references[i]
        while len(refs) < num_refs:
            refs.append("")
        new_references.append(refs)

    results = bart_scorer.multi_ref_score(hypothesis, new_references, agg="max", batch_size=4)

    # agg means aggregation, can be mean or max
    return results


def sentence_meteor_nltk(references, hypothesis):
    # check for empty lists
    references_, hypothesis_ = [], []
    print(len(references))
    print(len(hypothesis))
    for i, refs in enumerate(references):
        refs_ = [ref for ref in refs if ref.strip() != '']
        if len(refs_) > 0:
            references_.append([ref.split() for ref in refs_])
            hypothesis_.append(hypothesis[i].split())

    meteor = []
    print(len(references_))
    print(len(hypothesis_))
    for i in range(len(references_)):
        meteor.append(meteor_score(references_[i], hypothesis_[i]))
    return meteor


def bert_score_(references, hypothesis, lng='en'):
    logging.info('STARTING TO COMPUTE BERT SCORE...')
    print('STARTING TO COMPUTE BERT SCORE...')
    for i, refs in enumerate(references):
        references[i] = [ref for ref in refs if ref.strip() != '']

    try:
        P, R, F1 = score(hypothesis, references, lang=lng)
        logging.info('FINISHING TO COMPUTE BERT SCORE...')
        #     print('FINISHING TO COMPUTE BERT SCORE...')
        P, R, F1 = list(P), list(R), list(F1)

    except:
        P, R, F1 = 0, 0, 0
    return P, R, F1


def bleurt(references, hypothesis, num_refs, checkpoint="BLEURT20"):
    refs, cands = [], []
    for i, hyp in enumerate(hypothesis):
        for ref in references[i][:num_refs]:
            cands.append(hyp)
            refs.append(ref)
    print(len(refs))
    print(len(cands))

    scorer = bleurt_score.BleurtScorer("./BLEURT-20")
    scores = scorer.score(references=refs, candidates=cands)
    scores = [max(scores[i:i + num_refs]) for i in range(0, len(scores), num_refs)]
    return scores


def read_team_sample(filepath):
    rb = open(filepath, "r", encoding="utf-8")
    input_sentences = []
    for line in rb.readlines():
        sentence = line.strip()
        input_sentences.append(sentence)

    return input_sentences


def read_dataset(xmlfile, subset_ids):
    tree = ET.parse(xmlfile)

    # get root element
    root = tree.getroot()
    refs = []
    count = 1
    for child in root[0]:
        if subset_ids and count not in subset_ids:
            count += 1
            continue
        refs.append([])
        for entry in child.findall("lex"):
            # print(entry.text)
            refs[-1].append(entry.text)
        count += 1
    print(count)
    return refs


if __name__ == '__main__':
    parameters = json.load(open(sys.argv[1]))
    logging.info('FINISHING TO READ INPUTS...')

    metrics = parameters['metrics'].lower().split(',')
    metrics = "bleurt"
    files = [f.path for f in os.scandir(parameters['teams-samples'])]
    for file_name in files:
        print(file_name)
        references, references_tok, hypothesis, hypothesis_tok = parse(parameters['references'], file_name,
                                                                       parameters['num_references'])
        references_faith, references_faith_tok, hypothesis, hypothesis_tok = parse(parameters['references-faith'],
                                                                                   file_name, 1)
        method = file_name.split('/')[-1]
        logging.info('STARTING EVALUATION...')
        if 'bleu' in metrics:
            path = parameters['metrics-path'] + "/bleu/"
            if not exists(path):
                os.mkdir(path)
            wb = open(path + method, "w")
            score_b = sentence_bleu_nltk(references_tok, hypothesis_tok)
            for s in score_b:
                wb.write(str(s) + "\n")
            wb.close()

        if 'meteor' in metrics:
            path = parameters['metrics-path'] + "/meteor/"
            if not exists(path):
                os.mkdir(path)
            wb = open(path + method, "w")
            score_m = sentence_meteor_nltk(references_tok, hypothesis_tok)
            for s in score_m:
                wb.write(str(s) + "\n")
            wb.close()

        if 'bart' in metrics:
            path = parameters['metrics-path'] + "/bart/"
            if not exists(path):
                os.mkdir(path)
            wb = open(path + method, "w")
            print(len(hypothesis))
            scores = bart_score(references, hypothesis, parameters['num_references'])
            for s in scores:
                wb.write(str(s) + "\n")
            wb.close()

            path = parameters['metrics-path'] + "/bart-faith/"
            if not exists(path):
                os.mkdir(path)
            wb = open(path + method, "w")
            print(len(hypothesis))
            scores = bart_score(references_faith, hypothesis, 1)
            for s in scores:
                wb.write(str(s) + "\n")
            wb.close()

        if 'infolm' in metrics:
            path = parameters['metrics-path'] + "/infolm/"
            if not exists(path):
                os.mkdir(path)
            wb = open(path + method, "w")
            score_d = infolm_score(references, hypothesis)
            for s in score_d:
                wb.write(str(s) + "\n")
            wb.close()

        if 'bert' in metrics:
            path = parameters['metrics-path'] + "/bert-f1/"
            if not exists(path):
                os.mkdir(path)
            wb = open(path + method, "w")
            P, R, F1 = bert_score_(references, hypothesis, lng='en')
            for f in F1:
                wb.write(str(f.item()) + "\n")
            wb.close()

        if 'bleurt' in metrics:
            path = parameters['metrics-path'] + "/bleurt/"
            if not exists(path):
                os.mkdir(path)
            wb = open(path + method, "w")
            scores = bleurt(references, hypothesis, parameters['num_references'])
            for s in scores:
                wb.write(str(s) + "\n")
            wb.close()

        logging.info('FINISHING EVALUATION...')
