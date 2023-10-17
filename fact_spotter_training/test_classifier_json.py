import csv
import logging
import os
import torch
import json
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, SequentialSampler
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Most parts are similar to WikidataDataset
class PredicateTestDataset(Dataset):
    def __init__(self, json_path, data_tokenizer):
        self.data_path = json_path
        self.tokenizer = data_tokenizer
        self.data = []
        with open(self.data_path, 'r') as f:
            json_data = json.load(f)
            for json_id in json_data:
                self.data.append(json_data[json_id])
        print("Total samples = {}".format(len(self.data)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tmp_data = self.data[idx]
        tmp_seq = tmp_data[0]
        tmp_gt = tmp_data[1]
        return tmp_seq, tmp_gt


def testing(inf_model, inf_loader, tokenizer_inf, logger_inf, ):
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    with torch.no_grad():
        inf_model.eval()
        inf_iterator = tqdm(inf_loader, desc="Inference")
        all_gt = []
        all_pred = []
        for inf_seq, inf_gt in inf_iterator:
            gen_seq = [inf_seq[idx] for idx in range(0, len(inf_seq))]
            torch.cuda.empty_cache()
            tokenized_seq = tokenizer_inf(gen_seq)
            # pad each batch to same length
            tokenized_seq["input_ids"] = pad_sequence([torch.LongTensor(x) for x in tokenized_seq["input_ids"]],
                                                      padding_value=tokenizer_inf.pad_token_id, batch_first=True)
            tokenized_seq["attention_mask"] = pad_sequence([torch.LongTensor(x)
                                                            for x in tokenized_seq["attention_mask"]],
                                                           padding_value=0, batch_first=True)
            if torch.cuda.is_available():
                tokenized_seq["input_ids"] = tokenized_seq["input_ids"].to(torch.device("cuda"))
                tokenized_seq["attention_mask"] = tokenized_seq["attention_mask"].to(torch.device("cuda"))
            gen_prob = inf_model(input_ids=tokenized_seq["input_ids"], attention_mask=tokenized_seq["attention_mask"])
            # print(gt_prob)
            # print(torch.softmax(gt_prob.logits, dim=1, ))
            # print(gen_prob)
            # print(torch.softmax(gen_prob.logits, dim=1, ))
            gen_num = [1 if x[1] > .5 else 0 for x in torch.softmax(gen_prob.logits, dim=1, )]
            all_gt.extend(inf_gt.tolist())
            all_pred.extend(gen_num)
        accuracy = accuracy_score(all_gt, all_pred)
        conf = confusion_matrix(all_gt, all_pred, labels=[0, 1])
        f1 = f1_score(all_gt, all_pred, average="weighted")
        logger_inf.info("Accuracy: " + str(accuracy))
        logger_inf.info("F1: " + str(f1))
        logger_inf.info("Confusion Matrix:")
        logger_inf.info(str(conf))


class PDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, workers):
        sampler = SequentialSampler(dataset)
        super(PDataLoader, self).__init__(dataset, sampler=sampler, batch_size=batch_size, num_workers=workers)


if __name__ == '__main__':
    log_filename = "./cls_test_log.txt"
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO,
                        handlers=[logging.FileHandler(log_filename),
                                  logging.StreamHandler()])
    logger = logging.getLogger(__name__)
    tokenizer = AutoTokenizer.from_pretrained("google/electra-small-discriminator")
    eval_dataset = PredicateTestDataset("test_files/test_cls_webnlg17.json", tokenizer, )
    eval_loader = PDataLoader(eval_dataset, 1, 1, )
    model = AutoModelForSequenceClassification.from_pretrained("google/electra-small-discriminator", )
    # model_state_dict = torch.load("distill_bert0.8167039219670799.pt")
    # this is best model
    model_state_dict = torch.load("electra_webnlg_fix.5pred.25ent.25both.1drop0.9888319088319089.pt")
    model.load_state_dict(model_state_dict)
    if torch.cuda.is_available():
        model.to(torch.device("cuda"))
    testing(model, eval_loader, tokenizer, logger)
