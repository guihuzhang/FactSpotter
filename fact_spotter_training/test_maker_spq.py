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
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm, trange
from transformers import AutoTokenizer, AutoModelForSequenceClassification

os.environ["TOKENIZERS_PARALLELISM"] = "false"
sentence_trans = SentenceTransformer('all-MiniLM-L6-v2')


# Most parts are similar to WikidataDataset
class FixedDatasetMaker:
    def __init__(self, path_and_type, is_train, content_drop_percent, neg_pred_percent):
        self.content_drop_percent = content_drop_percent
        self.neg_pred_percent = neg_pred_percent
        self.lemma_model = spacy.load('en_core_web_sm')
        self.data_dict = []
        for each_data_and_type in path_and_type:
            with open(each_data_and_type[0], 'r') as f:
                self.data_dict.append([json.load(f), each_data_and_type[1]])
        self.is_train = is_train
        if is_train:
            random.seed(random.randint(0, sys.maxsize))
        else:
            random.seed(2023)
        self.add_bos_id = []
        self.prevent_predicates = ["class name", "entity name", "domains", "literal name"]
        self.property_predicates = ["class name", "entity name", "literal name"]
        self.predicate_list, self.entity_list, self.predicates_par_topic = self.get_all_predicates_and_entities()
        # for each_p in self.predicates_par_topic:
        #     print(each_p, self.predicates_par_topic[each_p])
        self.dataset_list, self.classes = self.make_dataset_list()
        print("Total samples = {}".format(len(self.dataset_list)))

    def __len__(self):
        return len(self.dataset_list)

    def get_all_predicates_and_entities(self):
        all_predicates = set()
        all_entities = set()
        predicates_par_topic = {}
        for each_data_and_type in self.data_dict:
            for each_record in each_data_and_type[0]:
                relation_id = 0
                for each_relation in each_record["kbs"]:
                    if relation_id == 0:
                        # because each simple_question only has one predicate useful
                        predicate_name = each_record["kbs"][each_relation][2][0][0]
                        predicate_domain = predicate_name.split(" . ")[0]
                        all_predicates.add(predicate_name)
                        if predicate_domain not in predicates_par_topic:
                            predicates_par_topic.update({predicate_domain: [predicate_name]})
                        else:
                            if predicate_name not in predicates_par_topic[predicate_domain]:
                                predicates_par_topic[predicate_domain].append(predicate_name)
                    else:
                        for each_po in each_record["kbs"][each_relation][2]:
                            tmp_entity = each_po[1].lower()
                            tmp_question = each_record["text"][0].lower()
                            if (tmp_entity in tmp_question) and (each_po[0] == "type . object . name" or
                                                                 each_po[0] == "common . topic . alias"):
                                all_entities.add(tmp_entity)
                    relation_id += 1
        return list(all_predicates), list(all_entities), predicates_par_topic

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

    def check_a_in_b(self, word_set_a, word_set_b):
        word_set_a = word_set_a.replace(" _ ", " ").replace(" . ", " ")
        lemma_a = set(x.lemma_ for x in self.lemma_model(word_set_a))
        lemma_b = set(x.lemma_ for x in self.lemma_model(word_set_b))
        for each_word in lemma_a:
            if each_word not in lemma_b:
                return False
        return True

    def make_dataset_list(self):
        list_dataset = []
        max_times = 0
        for each_data_and_type in self.data_dict:
            for each_record in each_data_and_type[0]:
                gt = each_record["text"]
                tmp_question = each_record["text"][0].lower()
                subject_id = ""
                spq_predicate = ""
                spq_entity = ""
                possible_entity_list = set()
                relation_id = 0
                for each_relation in each_record["kbs"]:
                    if relation_id == 0:
                        # because each simple_question only has one predicate useful
                        subject_id = each_record["kbs"][each_relation][0]
                        spq_predicate = each_record["kbs"][each_relation][2][0][0]
                    elif each_record["kbs"][each_relation][0] == subject_id:
                        for each_po in each_record["kbs"][each_relation][2]:
                            tmp_entity = each_po[1].lower()
                            if each_po[0] == "type . object . name" or each_po[0] == "common . topic . alias":
                                possible_entity_list.add(each_po[1])
                            if (tmp_entity in tmp_question) and (each_po[0] == "type . object . name" or
                                                                 each_po[0] == "common . topic . alias"):
                                spq_entity = tmp_entity
                    relation_id += 1
                list_dataset.append({"gt": gt, "current_entity": spq_entity, "current_predicate": spq_predicate,
                                     "all_possible_expression": ", ".join(possible_entity_list)})
        return list_dataset, max_times + 1

    def __getitem__(self, idx):
        tmp_data = self.dataset_list[idx]
        target = 1
        negative_sample = False
        negative_percent = .5
        negative_pred = False
        negative_pred_percent = self.neg_pred_percent
        hard_pred = False
        hard_pred_percent = .8
        drop_pred = False
        negative_entity = False
        drop_both_percent = .5
        drop_ent = False
        content_drop_percent = self.content_drop_percent
        drop_words = False
        random_erase_percent = .2
        if self.is_train:
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
        else:
            if idx % 2000 < 1000:
                negative_sample = True
                if idx % 500 < (500 * negative_pred_percent):
                    negative_pred = True
                    if idx % 5 < 4:
                        hard_pred = True
                    if idx % 50 < (50 * content_drop_percent):
                        drop_pred = True
                else:
                    negative_entity = True
                    if idx % 20 < (20 * content_drop_percent):
                        drop_ent = True
                    if idx % 100 < 50:
                        negative_pred = True
                        if idx % 5 < 4:
                            hard_pred = True
                        if idx % 50 < (50 * content_drop_percent):
                            drop_pred = True
                if (drop_ent or drop_pred) and (idx % 10 < 2):
                    drop_words = True
        # seed already fixed in _init_ for testing
        tmp_gt = random.choice(tmp_data["gt"])
        if negative_sample:
            if negative_pred:
                if drop_pred:
                    closest_ngram = self.choose_closest_ngram(tmp_gt, tmp_data["current_predicate"])
                    if closest_ngram in tmp_gt:
                        tmp_gt = " ".join(tmp_gt.replace(closest_ngram, "").split())
                        target = 0
                predicate_domain = tmp_data["current_predicate"].split(" . ")[0]
                new_predicate = random.choice(self.predicates_par_topic[predicate_domain])
                if not hard_pred or new_predicate == tmp_data["current_predicate"]:
                    new_predicate = random.choice(self.predicate_list)
                if new_predicate != tmp_data["current_predicate"]:
                    tmp_data["current_predicate"] = new_predicate
                    target = 0
            if negative_entity:
                if drop_ent:
                    replacing_entity = ""
                else:
                    replacing_entity = random.choice(self.entity_list)
                tmp_entity = tmp_data["current_entity"]
                if replacing_entity != tmp_entity and tmp_entity in tmp_gt and len(tmp_entity) > 2:
                    tmp_gt = " ".join(tmp_gt.replace(tmp_entity, replacing_entity).split())
                    target = 0
            if drop_words:
                new_sentence = self.random_drop_words(tmp_gt)
                if new_sentence != tmp_gt:
                    tmp_gt = new_sentence
                    target = 0
        return_string = "predicate: " + tmp_data["current_predicate"] + \
                        ", subject: " + tmp_data["all_possible_expression"] + ", sentence: " + tmp_gt
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
            model_result = model_train(input_ids=tokenized_seq["input_ids"],
                                       attention_mask=tokenized_seq["attention_mask"], labels=target).logits
            tmp_loss = loss_function(model_result, target)
            tmp_loss.backward()
            torch.nn.utils.clip_grad_norm_(model_train.parameters(), 1.)
        tmp_acc, tmp_f1 = evaluation(model_train, loader_eval, tokenizer_train, logger_train, epoch)
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
        f1 = f1_score(all_gt, all_pred, average="weighted")
        conf = confusion_matrix(all_gt, all_pred, labels=[0, 1])
        logger_inf.info("Epoch: " + str(epoch_num))
        logger_inf.info("Accuracy: " + str(accuracy))
        logger_inf.info("F1: " + str(f1))
        logger_inf.info("Confusion Matrix:")
        logger_inf.info(str(conf))
        return accuracy, f1


class PDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, workers, ):
        sampler = RandomSampler(dataset)
        super(PDataLoader, self).__init__(dataset, sampler=sampler, batch_size=batch_size, num_workers=workers)


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
    # train_list = [["spq_predicate_zero/train.json", "simple"]]
    # append_set = ["spq_predicate_zero/test.json", "simple"]
    # if append_test:
    #     train_list.append(append_set)
    # train_dataset = PredicateClassifierDataset(train_list, True, drop_percent, neg_pred_percent)
    test_dataset = FixedDatasetMaker([["../spq_predicate_zero/val.json", "simple"]], False,
                                     drop_percent, neg_pred_percent)
    test_dataset.output_dataset(output_dir)

    # train_loader = PDataLoader(train_dataset, b_size, workers, )
    # eval_loader = PDataLoader(eval_dataset, b_size, 1, )
    # model = AutoModelForSequenceClassification.from_pretrained(model_name)
    # # logger.info("Model Parameters:", sum(p.numel() for p in model.parameters()))
    #
    # loss_func = torch.nn.CrossEntropyLoss(weight=torch.Tensor([1, 1])).to(torch.device("cuda"))
    # if torch.cuda.is_available():
    #     model.to(torch.device("cuda"))
    # optimizer = AdamW(model.parameters(), lr=learning_rate, amsgrad=True)
    # scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=2)
    # train_func(logger, model, train_loader, eval_loader, optimizer,
    #            scheduler, tokenizer, epochs, loss_func, output_prefix)


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    fire.Fire(get_param_and_run)

