import json
import logging
import os
import random
import sys

import fire
import spacy
import torch
from nltk.corpus import wordnet
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import Dataset, DataLoader, RandomSampler
from tqdm import tqdm, trange
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from unidecode import unidecode

os.environ["TOKENIZERS_PARALLELISM"] = "false"
lemma_model = spacy.load('en_core_web_sm')
sentence_trans = SentenceTransformer('all-MiniLM-L6-v2')


# Most parts are similar to WikidataDataset
class FixedDatasetMaker:
    def __init__(self, path_and_type, predicate_jsons, antonym_json, is_train, content_drop_percent, neg_pred_percent):
        self.content_drop_percent = content_drop_percent
        self.neg_pred_percent = neg_pred_percent
        self.data_dict = []
        for each_data_and_type in path_and_type:
            with open(each_data_and_type[0], 'r') as f:
                self.data_dict.append([json.load(f), each_data_and_type[1]])
        self.predicate_dict = {}
        for each_json in predicate_jsons:
            with open(each_json, 'r') as f:
                self.predicate_dict.update(json.load(f))
        self.concept_antonym_dict = {}
        with open(antonym_json, "r") as f:
            self.concept_antonym_dict.update(json.load(f))
        self.is_train = is_train
        if is_train:
            random.seed(random.randint(0, sys.maxsize))
        else:
            random.seed(2023)
        self.add_bos_id = []
        self.prevent_predicates = ["class name", "entity name", "domains", "literal name", "sentence category",
                                   "[title]", "sentence data source"]
        self.prevent_entity = ["[TABLECONTEXT]"]
        self.predicate_list, self.entity_list = self.get_all_predicates_and_entities()
        # for each_p in self.predicates_par_topic:
        #     print(each_p, self.predicates_par_topic[each_p])
        self.dataset_list = self.make_dataset_list()
        print("Total samples = {}".format(len(self.dataset_list)))

    def __len__(self):
        return len(self.dataset_list)

    def get_all_predicates_and_entities(self):
        all_predicates = set()
        all_entities = set()
        for each_data_and_type in self.data_dict:
            for each_record in each_data_and_type[0]:
                for each_relation in each_record["kbs"]:
                    subject_name = each_record["kbs"][each_relation][0]
                    all_entities.add(subject_name.lower())
                    for each_po in each_record["kbs"][each_relation][2]:
                        predicate_name = each_po[0]
                        object_name = each_po[1]
                        all_predicates.add(predicate_name.lower())
                        all_entities.add(object_name.lower())
        return list(all_predicates), list(all_entities)

    @staticmethod
    def check_a_in_b(word_set_a, word_set_b):
        word_set_a = word_set_a.replace(" _ ", " ").replace(" . ", " ")
        lemma_a = set(x.lemma_ for x in lemma_model(word_set_a))
        lemma_b = set(x.lemma_ for x in lemma_model(word_set_b))
        for each_word in lemma_a:
            if each_word not in lemma_b:
                return False
        return True

    @staticmethod
    def choose_closest_ngram(gt_sentence, predicate):
        if predicate in gt_sentence:
            return predicate
        tmp_words = gt_sentence.split()
        max_ngram = 3 * len(predicate.split())
        ngrams_list = [' '.join(tmp_words[i:i + n]) for n in range(1, max_ngram + 1)
                       for i in range(len(tmp_words) - n + 1)]
        ngram_embeddings = sentence_trans.encode(ngrams_list, convert_to_tensor=True)
        predicate_embedding = sentence_trans.encode(predicate, convert_to_tensor=True)
        all_sim = util.cos_sim(predicate_embedding, ngram_embeddings)
        best_ngram = ngrams_list[all_sim.argmax()]
        return best_ngram

    @staticmethod
    def get_antonyms_wordnet(tmp_word):
        antonyms = []
        for syn in wordnet.synsets(tmp_word):
            for lemma in syn.lemmas():
                if lemma.antonyms():
                    antonyms.append(lemma.antonyms()[0].name().replace("_", " "))
        return antonyms

    def compose_antonym(self, candidate_phrase):
        all_possible_negatives = {candidate_phrase: 0}
        tmp_words = candidate_phrase.split()
        has_antonyms = 0
        for each_word in tmp_words:
            if each_word in self.concept_antonym_dict:
                antonyms_wordnet = set(self.get_antonyms_wordnet(each_word))
                antonyms_concept_net = set(self.concept_antonym_dict[each_word])
                if len(antonyms_wordnet) > 0:
                    tmp_antonyms = antonyms_wordnet
                else:
                    tmp_antonyms = antonyms_concept_net
                if len(tmp_antonyms) > 0:
                    has_antonyms += 1
                tmp_dict = {}
                for each_previous in all_possible_negatives:
                    for each_antonym in tmp_antonyms:
                        if len(each_antonym) > 1:
                            if each_antonym not in each_previous:
                                tmp_neg = each_previous.replace(each_word, each_antonym)
                                tmp_dict[tmp_neg] = all_possible_negatives[each_previous] + 1
                all_possible_negatives.update(tmp_dict)
        expected_change = has_antonyms / 2
        tmp_list = [tmp_key for tmp_key in all_possible_negatives if all_possible_negatives[tmp_key] > expected_change]
        return tmp_list

    @staticmethod
    def random_drop_words(dropping_sentence):
        tmp_words = dropping_sentence.split()
        new_words = []
        for each_word in tmp_words:
            if random.random() > .5:
                new_words.append(each_word)
        return " ".join(new_words)

    @staticmethod
    def random_repeat_words(repeating_sentence):
        tmp_words = repeating_sentence.split()
        tmp_len = len(tmp_words)
        repeating_time = tmp_len // 10 + 1
        if repeating_time <= tmp_len:
            words2repeat = random.sample(tmp_words, repeating_time)
            for word_r in words2repeat:
                tmp_pos = random.randint(0, tmp_len)
                tmp_words.insert(tmp_pos, word_r)
        return " ".join(tmp_words)

    @staticmethod
    def random_shuffle_words(random_shuffle_sentence):
        tmp_words = random_shuffle_sentence.split()
        random.shuffle(tmp_words)
        return " ".join(tmp_words)

    @staticmethod
    def webnlg_sentences_format_correction(input_sentences):
        output_sent = []
        for each_sent in input_sentences:
            output_sent.append(unidecode(each_sent).lower().strip())
        return output_sent

    @staticmethod
    def webnlg_entity_format_correction(entity_letters):
        if "(" in entity_letters:
            entity_letters = entity_letters.split("(")[0]
        if "," in entity_letters:
            entity_letters = entity_letters.split(",")[0]
        return unidecode(entity_letters).replace("language", "").lower().strip()

    def make_dataset_list(self):
        # print(entry['kbs'])
        list_dataset = []
        # entity_count = 0
        for each_data_and_type in self.data_dict:
            for each_record in each_data_and_type[0]:
                predicate_dict = set()
                entity_dict = set()
                category = ""
                triple_list = []
                for each_relation in each_record["kbs"]:
                    for each_po in each_record["kbs"][each_relation][2]:
                        if each_po[0] == "sentence category":
                            category = each_po[1]
                for each_relation in each_record["kbs"]:
                    subject_name = unidecode(each_record["kbs"][each_relation][0]).lower()
                    for each_po in each_record["kbs"][each_relation][2]:
                        predicate_name = each_po[0].lower()
                        if predicate_name not in self.prevent_predicates:
                            predicate_dict.add(predicate_name)
                            if subject_name not in self.prevent_entity:
                                entity_dict.add(subject_name)
                            object_name = unidecode(each_po[1]).lower()
                            if object_name not in self.prevent_entity:
                                entity_dict.add(object_name)
                            triple_list.append([subject_name, predicate_name, object_name])
                    for gt in each_record["text"]:
                        for each_triple in triple_list:
                            list_dataset.append({"gt": [unidecode(gt.lower().strip("?!. "))],
                                                 "all_predicates": predicate_dict, "all_entities": entity_dict,
                                                 "current_triple": each_triple, "category": category})

        return list_dataset

    def __getitem__(self, idx):
        tmp_data = self.dataset_list[idx]
        target = 1
        training_entity_aug = False
        augment_percent = .5
        negative_sample = False
        negative_percent = .5
        negative_pred = False
        negative_pred_percent = self.neg_pred_percent
        hard_pred = False
        hard_pred_percent = .8
        drop_pred = False
        negative_entity = False
        drop_both_percent = .5
        change_sbj = False
        reverse_so = False
        reverse_so_percent = .1
        negative_subject_percent = .5
        drop_ent = False
        content_drop_percent = self.content_drop_percent
        drop_words = False
        shuffle_words = False
        repeat_words = False
        random_erase_percent = .1
        random_shuffle_percent = .1
        random_repeat_percent = .1
        if self.is_train:
            # for training, randomly replace entity and pred
            augment_float = random.random()
            if augment_float < augment_percent:
                training_entity_aug = True
            negative_float = random.random()
            if negative_float < negative_percent:
                negative_sample = True
                pred_float = random.random()
                if pred_float <= negative_pred_percent:
                    negative_pred = True
                    hard_float = random.random()
                    if hard_float <= hard_pred_percent:
                        hard_pred = True
                    pred_drop_float = random.random()
                    if pred_drop_float <= content_drop_percent:
                        drop_pred = True
                else:
                    negative_entity = True
                    reverse_float = random.random()
                    if reverse_float < reverse_so_percent:
                        reverse_so = True
                    else:
                        so_float = random.random()
                        if so_float < negative_subject_percent:
                            change_sbj = True
                        ent_drop_float = random.random()
                        if ent_drop_float <= content_drop_percent:
                            drop_ent = True
                        # give half prob neg pred for neg ent
                        pred_float = random.random()
                        if pred_float <= drop_both_percent:
                            negative_pred = True
                            hard_float = random.random()
                            if hard_float <= hard_pred_percent:
                                hard_pred = True
                            pred_drop_float = random.random()
                            if pred_drop_float <= content_drop_percent:
                                drop_pred = True
                if drop_ent or drop_pred:
                    random_drop_float = random.random()
                    if random_drop_float <= random_erase_percent:
                        drop_words = True
                    random_shuffle_float = random.random()
                    if random_shuffle_float <= random_shuffle_percent:
                        shuffle_words = True
                    random_repeat_float = random.random()
                    if random_repeat_float <= random_repeat_percent:
                        repeat_words = True
        else:
            if idx % 2000 < 1000:
                negative_sample = True
                if idx % 1000 < (1000 * negative_pred_percent):
                    negative_pred = True
                    if idx % 5 < 4:
                        hard_pred = True
                    if idx % 50 < (50 * content_drop_percent):
                        drop_pred = True
                else:
                    negative_entity = True
                    if idx % 2 < 1:
                        change_sbj = True
                    if idx % 20 < (20 * content_drop_percent):
                        drop_ent = True
                    if idx % 100 < 50:
                        negative_pred = True
                        if idx % 5 < 4:
                            hard_pred = True
                        if idx % 50 < (50 * content_drop_percent):
                            drop_pred = True
                    if idx % 200 < 20:
                        reverse_so = True
                if drop_ent or drop_pred:
                    if idx % 10 < 1:
                        drop_words = True
                    elif idx % 10 < 2:
                        shuffle_words = True
                    elif idx % 10 < 3:
                        repeat_words = True
        # seed already fixed in _init_ for testing
        tmp_data["gt"] = " ".join(random.choice(tmp_data["gt"]).strip(" .?").lower().split())
        tmp_sub, tmp_pred, tmp_obj = tmp_data["current_triple"]
        # randomly change entity name for data augmentation
        if training_entity_aug:
            for each_entity in tmp_data["all_entities"]:
                tmp_entity = self.webnlg_entity_format_correction(each_entity)
                if tmp_entity in tmp_data["gt"] and len(tmp_entity) > 2:
                    replacing_entity = random.choice(self.entity_list)
                    if each_entity == tmp_sub:
                        tmp_sub = replacing_entity
                    if each_entity == tmp_obj:
                        tmp_obj = replacing_entity
                    tmp_data["gt"] = tmp_data["gt"].replace(tmp_entity, replacing_entity)
        if negative_sample:
            if negative_pred:
                if drop_pred:
                    closest_ngram = self.choose_closest_ngram(tmp_data["gt"], tmp_pred)
                    if closest_ngram in tmp_data["gt"]:
                        tmp_data["gt"] = " ".join(tmp_data["gt"].replace(closest_ngram, "").split())
                        target = 0
                predicate_domain = tmp_data["category"]
                negative_selection_set = self.predicate_dict[predicate_domain]
                # antonyms = self.compose_antonym(tmp_pred)
                # if len(antonyms) > 0:
                #     negative_selection_set.extend(antonyms)
                new_predicate = random.choice(negative_selection_set).lower()
                if not hard_pred or new_predicate == tmp_pred:
                    new_predicate = random.choice(self.predicate_list)
                # if predicate is different, then make negative
                if new_predicate != tmp_pred:
                    tmp_pred = new_predicate
                    target = 0
            if negative_entity:
                if reverse_so and tmp_sub != tmp_obj and tmp_sub in tmp_data["gt"] and tmp_obj in tmp_data["gt"]:
                    tmp_data["gt"] = " ".join(tmp_data["gt"].replace(tmp_sub, tmp_obj.upper()).replace(
                        tmp_obj, tmp_sub.upper()).split()).lower()
                    target = 0
                else:
                    if drop_ent:
                        replacing_entity = ""
                    else:
                        replacing_entity = random.choice(self.entity_list)
                    # change subject
                    if change_sbj:
                        if replacing_entity != tmp_sub and tmp_sub in tmp_data["gt"] and len(tmp_sub) > 2:
                            tmp_data["gt"] = " ".join(tmp_data["gt"].replace(tmp_sub, replacing_entity).split())
                            target = 0
                    elif replacing_entity != tmp_obj and tmp_obj in tmp_data["gt"] and len(tmp_obj) > 2:
                        # change object
                        tmp_data["gt"] = " ".join(tmp_data["gt"].replace(tmp_obj, replacing_entity).split())
                        target = 0
            if repeat_words:
                new_sentence = self.random_repeat_words(tmp_data["gt"])
                if new_sentence != tmp_data["gt"]:
                    tmp_data["gt"] = new_sentence
                    target = 0
            if shuffle_words:
                new_sentence = self.random_shuffle_words(tmp_data["gt"])
                if new_sentence != tmp_data["gt"]:
                    tmp_data["gt"] = new_sentence
                    target = 0
            if drop_words:
                new_sentence = self.random_drop_words(tmp_data["gt"])
                if new_sentence != tmp_data["gt"]:
                    tmp_data["gt"] = new_sentence
                    target = 0
        return_string = "predicate: " + tmp_pred + ", subject: " + tmp_sub + \
                        ", object: " + tmp_obj + ", sentence: " + tmp_data["gt"]
        return return_string, target

    def output_dataset(self, output_directory):
        output_list = {}
        for did in range(0, self.__len__()):
            tmp_str, tmp_tgt = self.__getitem__(did)
            output_list[did] = [tmp_str, tmp_tgt]
        with open(output_directory, 'w') as f:
            json.dump(output_list, f, indent=2)


def train_func(logger_train, model_train, loader_train, loader_eval, optimiser_train, scheduler_train, tokenizer_train,
               train_epochs, loss_function, train_prefix):
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    best_f1 = -1
    train_iterator = trange(int(train_epochs), desc="Epoch")
    logger_train.info("Starting training!")
    for epoch in train_iterator:
        epoch_iterator = tqdm(loader_train, desc="Train")
        model_train.train()
        logger_train.info("Learning Rate:" + str(scheduler_train.get_last_lr()))
        for seq, target in epoch_iterator:
            torch.cuda.empty_cache()
            tokenized_seq = tokenizer_train(seq, truncation=True)
            # pad each batch to same length
            tokenized_seq["input_ids"] = pad_sequence([torch.LongTensor(x) for x in tokenized_seq["input_ids"]],
                                                      padding_value=tokenizer_train.pad_token_id, batch_first=True)
            tokenized_seq["attention_mask"] = pad_sequence([torch.LongTensor(x) for x in
                                                            tokenized_seq["attention_mask"]],
                                                           padding_value=0, batch_first=True)
            if torch.cuda.is_available():
                target = target.to(torch.device("cuda"))
                tokenized_seq["input_ids"] = tokenized_seq["input_ids"].to(torch.device("cuda"))
                tokenized_seq["attention_mask"] = tokenized_seq["attention_mask"].to(torch.device("cuda"))
            optimiser_train.step()
            model_train.zero_grad()
            optimiser_train.zero_grad()
            # print(target, tokenized_seq)
            # print(tokenized_seq["input_ids"].device())
            # print(tokenized_seq["attention_mask"].device())
            # print(target.device())
            model_result = model_train(input_ids=tokenized_seq["input_ids"],
                                       attention_mask=tokenized_seq["attention_mask"], labels=target).logits
            # tmp_loss = model_train(input_ids=tokenized_seq["input_ids"],
            # attention_mask=tokenized_seq["attention_mask"],
            # labels=target).loss
            tmp_loss = loss_function(model_result, target)
            tmp_loss.backward()
            torch.nn.utils.clip_grad_norm_(model_train.parameters(), 1.)
        tmp_acc, tmp_f1, tmp_conf = evaluation(model_train, loader_eval, tokenizer_train, logger_train, epoch)
        scheduler_train.step()
        if tmp_f1 > best_f1:
            best_f1 = tmp_f1
            logger_train.info("Best F1!")
            torch.save(model_train.state_dict(), train_prefix + str(tmp_acc) + ".pt")


def evaluation(inf_model, inf_loader, tokenizer_inf, logger_inf, epoch_num):
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    with torch.no_grad():
        inf_model.eval()
        inf_iterator = tqdm(inf_loader, desc="Inference")
        all_gt = []
        all_pred = []
        for seq, target in inf_iterator:
            torch.cuda.empty_cache()
            tokenized_seq = tokenizer_inf(seq)
            # pad each batch to same length
            tokenized_seq["input_ids"] = pad_sequence([torch.LongTensor(x) for x in tokenized_seq["input_ids"]],
                                                      padding_value=tokenizer_inf.pad_token_id, batch_first=True)
            tokenized_seq["attention_mask"] = pad_sequence([torch.LongTensor(x)
                                                            for x in tokenized_seq["attention_mask"]],
                                                           padding_value=0, batch_first=True)
            if torch.cuda.is_available():
                target = target.to(torch.device("cuda"))
                tokenized_seq["input_ids"] = tokenized_seq["input_ids"].to(torch.device("cuda"))
                tokenized_seq["attention_mask"] = tokenized_seq["attention_mask"].to(torch.device("cuda"))
            tmp_pred = inf_model(input_ids=tokenized_seq["input_ids"], attention_mask=tokenized_seq["attention_mask"])
            _, pred_index = tmp_pred.logits.max(dim=1, )
            all_pred.extend(pred_index.tolist())
            all_gt.extend(target.tolist())
            # print(seq, target, pred_index)
        # prevent disgusting code that everytime needs downloading
        accuracy = accuracy_score(all_gt, all_pred)
        conf = confusion_matrix(all_gt, all_pred, labels=[0, 1])
        f1 = f1_score(all_gt, all_pred, average="weighted")
        logger_inf.info("Epoch: " + str(epoch_num))
        logger_inf.info("Accuracy: " + str(accuracy))
        logger_inf.info("F1: " + str(f1))
        logger_inf.info("Confusion Matrix:")
        logger_inf.info(str(conf))
        return accuracy, f1, conf


class PDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, workers):
        sampler = RandomSampler(dataset)
        super(PDataLoader, self).__init__(dataset, sampler=sampler, batch_size=batch_size, num_workers=workers,
                                          pin_memory=True)


def get_param_and_run(drop_percent, neg_pred_percent, output_dir):

    log_filename = "./cls_train_log.txt"
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO,
                        handlers=[logging.FileHandler(log_filename),
                                  logging.StreamHandler()])
    # logger = logging.getLogger(__name__)
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # if "const" in dataset_name:
    #     train_list = [["webnlg_const_kun/train.json", "webnlg17"]]
    #     append_set = ["webnlg_const_kun/test.json", "webnlg17"]
    # else:
    #     train_list = [["webnlg_kun/train.json", "webnlg17"]]
    #     append_set = ["webnlg_kun/test.json", "webnlg17"]
    # if append_test:
    #     train_list.append(append_set)
    # print(model_name, learning_rate, b_size, epochs, output_prefix, drop_percent, neg_pred_percent)
    # print(train_list)
    # ["webnlg_kun/train.json", "webnlg17"],  ["webnlg_kun/test.json", "webnlg17"], ],
    # ["webnlg_kun/train.json", "webnlg17"]],
    # ["augment_dataset/SimpleQ_no_zero_predicate.json", "simple"],
    # ["augment_dataset/GraphQ_no_zero_predicate.json", "grail"],
    # ["grail_dataset/train.json", "grail"]
    # train_dataset = PredicateClassifierDataset(train_list,
    #                                            ["webnlg_kun/train_pred_cls.json", "webnlg_kun/test_pred_cls.json"],
    #                                            "antonyms.json", True, drop_percent, neg_pred_percent)
    test_dataset = FixedDatasetMaker([["../webnlg20/test.json", "webnlg17"]],
                                     ["../webnlg20/test_pred_cls.json"],
                                     "antonyms.json", False, drop_percent, neg_pred_percent)
    test_dataset.output_dataset(output_dir)
    # train_loader = PDataLoader(train_dataset, b_size, workers, )
    # eval_loader = PDataLoader(eval_dataset, b_size, 1, )
    # model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2, ignore_mismatched_sizes=True)
    #
    # loss_func = torch.nn.CrossEntropyLoss()
    # if torch.cuda.is_available():
    #     model.to(torch.device("cuda"))
    # optimizer = AdamW(model.parameters(), lr=learning_rate, amsgrad=True)
    # scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=2)
    # train_func(logger, model, train_loader, eval_loader, optimizer,
    #            scheduler, tokenizer, epochs, loss_func, output_prefix)


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    fire.Fire(get_param_and_run)
