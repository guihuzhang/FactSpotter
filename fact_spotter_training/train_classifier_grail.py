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
lemma_model = spacy.load('en_core_web_sm')
sentence_trans = SentenceTransformer('all-MiniLM-L6-v2')


# Most parts are similar to WikidataDataset
class PredicateClassifierDataset(Dataset):
    def __init__(self, path_and_type, antonym_json, is_train, content_drop_percent, neg_pred_percent):
        self.content_drop_percent = content_drop_percent
        self.neg_pred_percent = neg_pred_percent
        self.data_dict = []
        for each_data_and_type in path_and_type:
            with open(each_data_and_type[0], 'r') as f:
                self.data_dict.append([json.load(f), each_data_and_type[1]])
        self.is_train = is_train
        self.concept_antonym_dict = {}
        with open(antonym_json, "r") as f:
            self.concept_antonym_dict.update(json.load(f))
        if is_train:
            random.seed(random.randint(0, sys.maxsize))
        else:
            random.seed(2023)
        self.add_bos_id = []
        self.prevent_predicates = ["class name", "entity name", "domains", "literal name"]
        self.property_predicates = ["class name", "entity name", "literal name"]
        self.predicate_list, self.entity_list, self.domain_predicate_dict, self.predicate_domain_dict = \
            self.get_all_predicates_and_entities()
        self.dataset_list = self.make_dataset_list()
        print("Total samples = {}".format(len(self.dataset_list)))

    def __len__(self):
        return len(self.dataset_list)

    def get_all_predicates_and_entities(self):
        all_predicates = set()
        all_entities = set()
        domain_predicate_dict = {}
        predicate_domain_dict = {}
        for each_data_and_type in self.data_dict:
            for each_record in each_data_and_type[0]:
                for each_relation in each_record["kbs"]:
                    for each_po in each_record["kbs"][each_relation][2]:
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
                                predicate_domain = function_and_name[0]
                            else:
                                tmp_predicate = each_po[0].lower()
                                predicate_domain = tmp_predicate.split(" : ")[1].lower().split(" . ")[0]
                            tmp_predicate = tmp_predicate.replace(" : ", " < name > ")
                            all_predicates.add(tmp_predicate)
                            if predicate_domain not in domain_predicate_dict:
                                domain_predicate_dict.update({predicate_domain: [tmp_predicate]})
                            elif tmp_predicate not in domain_predicate_dict[predicate_domain]:
                                domain_predicate_dict[predicate_domain].append(tmp_predicate)
                            predicate_domain_dict[tmp_predicate] = predicate_domain
                        if each_po[0] == "entity name" or each_po[0] == "literal name":
                            all_entities.add(each_po[1].split(" : ")[0].lower())
        return list(all_predicates), list(all_entities), domain_predicate_dict, predicate_domain_dict

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
        tmp_predicate = predicate.split(" < name > ")[0]
        if tmp_predicate in gt_sentence:
            return tmp_predicate
        tmp_words = gt_sentence.split()
        max_ngram = 3 * len(tmp_predicate.split())
        ngrams_list = [' '.join(tmp_words[i:i + n]) for n in range(1, max_ngram + 1)
                       for i in range(len(tmp_words) - n + 1)]
        ngram_embeddings = sentence_trans.encode(ngrams_list, convert_to_tensor=True)
        predicate_embedding = sentence_trans.encode(tmp_predicate, convert_to_tensor=True)
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
    def get_antonyms_wordnet(tmp_word):
        antonyms = []
        for syn in wordnet.synsets(tmp_word):
            for lemma in syn.lemmas():
                if lemma.antonyms():
                    antonyms.append(lemma.antonyms()[0].name().replace("_", " "))
        return antonyms

    def compose_antonym(self, candidate_phrase):
        tmp_name = candidate_phrase.split(" < name > ")[0]
        all_possible_negatives = {candidate_phrase: 0}
        tmp_words = tmp_name.split()
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

    def make_dataset_list(self):
        # print(entry['kbs'])
        list_dataset = []
        for each_data_and_type in self.data_dict:
            for each_record in each_data_and_type[0]:
                gt = each_record["text"]
                triple_list = []
                entity_list = []
                entity_dict = {}
                predicate_list = set()
                for each_relation in each_record["kbs"]:
                    tmp_subject = each_record["kbs"][each_relation][0]
                    for each_po in each_record["kbs"][each_relation][2]:
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
                            predicate_list.add(tmp_predicate)
                        if each_po[0] == "entity name" or each_po[0] == "literal name":
                            entity_name = each_po[1].split(" : ")[0].lower()
                            if entity_name not in entity_list:
                                entity_list.append(entity_name)
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
                for each_triple in triple_list:
                    list_dataset.append({"gt": gt, "all_predicates": predicate_list, "all_entities": entity_list,
                                         "current_triple": each_triple, })
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
            if idx % 1000 < 500:
                negative_sample = True
                if idx % 500 < (500 * negative_pred_percent):
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
        # augment data by replacing entity names
        if training_entity_aug:
            for each_entity in tmp_data["all_entities"]:
                # randomly change entity name for data augmentation
                if each_entity in tmp_data["gt"] and len(each_entity) > 2:
                    tmp_sub_and_class = tmp_sub.split("< class >")
                    tmp_obj_and_class = tmp_obj.split("< class >")
                    replacing_entity = random.choice(self.entity_list)
                    no_change_flag = False
                    if each_entity in tmp_sub and len(tmp_sub_and_class) > 1:
                        tmp_sub_name = tmp_sub_and_class[0].strip()
                        sub_class = tmp_sub_and_class[1].strip()
                        if self.check_a_in_b(tmp_sub_name, sub_class):
                            no_change_flag = True
                        else:
                            tmp_sub = tmp_sub.replace(each_entity, replacing_entity)
                    if each_entity in tmp_obj and len(tmp_obj_and_class) > 1:
                        tmp_obj_name = tmp_obj_and_class[0].strip()
                        obj_class = tmp_obj_and_class[1].strip()
                        if self.check_a_in_b(tmp_obj_name, obj_class):
                            no_change_flag = True
                        else:
                            tmp_obj = tmp_obj.replace(each_entity, replacing_entity)
                    if not no_change_flag:
                        tmp_data["gt"] = tmp_data["gt"].replace(each_entity, replacing_entity)
        tmp_sub_and_class = tmp_sub.split("< class >")
        tmp_obj_and_class = tmp_obj.split("< class >")
        tmp_sub_name = tmp_sub_and_class[0].strip()
        tmp_obj_name = tmp_obj_and_class[0].strip()
        # choose negative predicates
        if negative_sample:
            if negative_pred:
                if drop_pred:
                    closest_ngram = self.choose_closest_ngram(tmp_data["gt"], tmp_pred)
                    if closest_ngram in tmp_data["gt"]:
                        tmp_data["gt"] = " ".join(tmp_data["gt"].replace(closest_ngram, "").split())
                        target = 0
                predicate_domain = self.predicate_domain_dict[tmp_pred]
                negative_selection_set = self.domain_predicate_dict[predicate_domain]
                antonyms = self.compose_antonym(tmp_pred)
                if len(antonyms) > 0:
                    negative_selection_set.extend(antonyms)
                new_predicate = random.choice(negative_selection_set).lower()
                if not hard_pred or new_predicate == tmp_pred:
                    new_predicate = random.choice(self.predicate_list)
                if new_predicate != tmp_pred:  # and new_predicate not in tmp_data["all_predicates"]:
                    tmp_pred = new_predicate
                    target = 0
            if negative_entity:
                if drop_ent:
                    replacing_entity = ""
                else:
                    replacing_entity = random.choice(self.entity_list)
                if change_sbj:
                    if len(tmp_sub_and_class) > 1 and len(tmp_sub_name) > 2:
                        # if entity changes, then make negative
                        if replacing_entity != tmp_sub_name and tmp_sub_name in tmp_data["gt"]:
                            tmp_data["gt"] = tmp_data["gt"].replace(tmp_sub_name, replacing_entity)
                            target = 0
                elif len(tmp_obj_and_class) > 1 and len(tmp_obj_name) > 2:
                    if replacing_entity != tmp_obj_name and tmp_obj_name in tmp_data["gt"]:
                        tmp_data["gt"] = tmp_data["gt"].replace(tmp_obj_name, replacing_entity)
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
                old_sentence = tmp_data["gt"]
                tmp_data["gt"] = self.random_drop_words(tmp_data["gt"])
                if tmp_data["gt"] != old_sentence:
                    target = 0
        return_string = "predicate: " + tmp_pred + ", subject: " + tmp_sub + \
                        ", object: " + tmp_obj + ", sentence: " + tmp_data["gt"]
        return return_string, target


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
        conf = confusion_matrix(all_gt, all_pred, labels=[0, 1])
        f1 = f1_score(all_gt, all_pred, average="weighted")
        logger_inf.info("Epoch: " + str(epoch_num))
        logger_inf.info("Accuracy: " + str(accuracy))
        logger_inf.info("F1: " + str(f1))
        logger_inf.info("Confusion Matrix:")
        logger_inf.info(str(conf))
        return accuracy, f1


class PDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, workers):
        sampler = SequentialSampler(dataset)
        super(PDataLoader, self).__init__(dataset, sampler=sampler, batch_size=batch_size, num_workers=workers,
                                          pin_memory=True)


def get_param_and_run(model_name, learning_rate, b_size, workers, epochs, append_test, output_prefix, drop_percent,
                      neg_pred_percent):
    log_filename = "./cls_train_log.txt"
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO,
                        handlers=[logging.FileHandler(log_filename),
                                  logging.StreamHandler()])
    logger = logging.getLogger(__name__)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_list = [
        # ["augment_dataset/GraphQ_no_zero_predicate.json", "grail"],
        ["../grail_new/train.json", "grail"]]
    append_set = ["../grail_new/test.json", "grail"]
    if append_test:
        train_list.append(append_set)
    # print(model_name, learning_rate, b_size, epochs, output_prefix, drop_percent, neg_pred_percent)
    print(train_list)
    train_dataset = PredicateClassifierDataset(train_list, "antonyms.json",  True, drop_percent, neg_pred_percent)
    eval_dataset = PredicateClassifierDataset([["../grail_new/val.json", "grail"]], "antonyms.json",
                                              False, drop_percent, neg_pred_percent)
    train_loader = PDataLoader(train_dataset, b_size, workers)
    eval_loader = PDataLoader(eval_dataset, b_size, 1)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    # logger.info("Model Parameters:", sum(p.numel() for p in model.parameters()))

    loss_func = torch.nn.CrossEntropyLoss(weight=torch.Tensor([1, 1])).to(torch.device("cuda"))
    if torch.cuda.is_available():
        model.to(torch.device("cuda"))
    optimizer = AdamW(model.parameters(), lr=learning_rate, amsgrad=True)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=2)
    train_func(logger, model, train_loader, eval_loader, optimizer,
               scheduler, tokenizer, epochs, loss_func, output_prefix)


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    fire.Fire(get_param_and_run)
