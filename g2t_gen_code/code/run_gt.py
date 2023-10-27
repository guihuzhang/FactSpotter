import os
import numpy as np
import torch
import random
from transformers import BartTokenizer, T5Tokenizer, GPT2Tokenizer, PhrasalConstraint, AutoTokenizer
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR
from modeling_bart_new import MyBartForConditionalGeneration as MyBart
from constrained_t5joint import MyT5ForConditionalGeneration as MyT5
from transformers import T5ForConditionalGeneration, BartForConditionalGeneration, AutoModelForSequenceClassification
from torch.nn.utils.rnn import pad_sequence

from modeling_bart import MyBartPretrain
from modeling_t5 import MyT5Pretrain
from modeling_kgpt import TransformerDecoder, GatedTransformerDecoder, GraphGatedTransformerDecoder, load_my_state_dict
from constrain_generate_util import generate

from bert_score import score
from data import WikidataDataset, WikidataDataLoader, WebNLGDataLoader, WebNLGDataset, DownStreamDataset
from data import evaluate_bleu
from tqdm import tqdm, trange
import json
from types import SimpleNamespace


def run(args, logger):
    # Initialise classification model
    cls_tokenizer = ""
    cls_model = ""
    if args.cls_model != "":
        cls_tokenizer = AutoTokenizer.from_pretrained(args.cls_model)
        cls_model = AutoModelForSequenceClassification.from_pretrained(args.cls_model)
        cls_state_dict = torch.load(args.cls_model_path)
        cls_model.load_state_dict(cls_state_dict)
        if torch.cuda.is_available():
            cls_model.to(torch.device("cuda"))
        cls_model.eval()
    cls_start_step = args.cls_start_step
    cls_threshold = args.cls_threshold
    promotion_weight = args.promotion_weight
    # Initialize tokenizer
    if "bart" in args.model_name:
        tokenizer = BartTokenizer.from_pretrained(args.tokenizer_path)
    elif "kgpt" in args.model_name:
        tokenizer = GPT2Tokenizer.from_pretrained(args.tokenizer_path)
        #  kgpt should load config outside
        knowledge_config_path = os.path.join(args.tokenizer_path, 'knowledge_config.json')
        with open(knowledge_config_path, 'r') as f:
            knowledge_config = json.load(f)
            config = SimpleNamespace(**knowledge_config)
            print(config)
    else:
        tokenizer = T5Tokenizer.from_pretrained(args.tokenizer_path)

    if args.do_pretrain:
        # Pretrain on kgtext
        with open(args.knowledge_file + '.json', 'r') as f:
            kg_data = json.load(f)
        train_dataset = WikidataDataset(logger, args, args.train_file, kg_data, tokenizer, "train")
        dev_dataset = WikidataDataset(logger, args, args.predict_file, kg_data, tokenizer, "val")
        train_dataloader = WikidataDataLoader(args, train_dataset, "train")
        dev_dataloader = WikidataDataLoader(args, dev_dataset, "dev")
    else:
        # Finetune on webnlg17 / webquestions / pathquestions
        if "kgpt" in args.model_name:
            # K: must specify encoding method
            if "seq" in args.model_name:
                tmp_enc = "sequence"
            else:
                tmp_enc = "graph"
            train_dataset = DownStreamDataset(args.train_file + ".json", tokenizer, max_entity=12, max_fact=12,
                                              max_enc_len=args.max_input_length,
                                              max_dec_len=args.max_output_length, encoder=tmp_enc)
            dev_dataset = DownStreamDataset(args.predict_file + ".json", tokenizer, max_entity=12, max_fact=12,
                                            max_enc_len=args.max_input_length,
                                            max_dec_len=args.max_output_length, encoder=tmp_enc)
        else:
            # JointGT dataset
            train_dataset = WebNLGDataset(logger, args, args.train_file, tokenizer, "train")
            dev_dataset = WebNLGDataset(logger, args, args.predict_file, tokenizer, "val")
        train_dataloader = WebNLGDataLoader(args, train_dataset, "train")
        dev_dataloader = WebNLGDataLoader(args, dev_dataset, "dev")

    if args.do_train:
        # Load model parameters
        if args.do_pretrain:
            model = MyBartPretrain.from_pretrained(args.model_path) if "bart-thu" in args.model_name \
                else MyT5Pretrain.from_pretrained(args.model_path)
        # K: fixing bug of THU model loading
        elif "bart-thu" in args.model_name:
            print("model is JointGT Bart!")
            model = MyBart.from_pretrained(args.model_path)
            # if cls exist, override generation function
        elif "t5-thu" in args.model_name:
            print("model is JointGT T5!")
            model = MyT5.from_pretrained(args.model_path)
        elif "bart" in args.model_name:
            print("model is normal Bart!")
            model = BartForConditionalGeneration.from_pretrained(args.model_path)
        elif "t5" in args.model_name:
            print("model is normal T5!")
            model = T5ForConditionalGeneration.from_pretrained(args.model_path)
        #  add definition of KGPT models
        elif "seq" in args.model_name:
            model = GatedTransformerDecoder(config, 8, 6)
        elif "graph" in args.model_name:
            model = GraphGatedTransformerDecoder(config, 8, 6)
        else:
            model = TransformerDecoder(config, 8, 6)
        if "kgpt" not in args.model_name:
            print('model parameters: ', model.num_parameters())
        # K: because KGPT are just torch models, not transformer models, so the loading is different
        if "kgpt" in args.model_name or args.n_gpu > 1:
            model = torch.nn.DataParallel(model)
        # loading should be after DP
        if "kgpt" in args.model_name:
            reloaded = torch.load(args.model_path)
            load_my_state_dict(model, reloaded)
        # if "bart" in args.model_name or "t5" in args.model_name:
        #     model.decode_tokenizer = tokenizer
        if torch.cuda.is_available():
            model.to(torch.device("cuda"))
        # load optimiser for training
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
        if "kgpt" not in args.model_name:
            no_decay = ['bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                 'weight_decay': args.weight_decay},
                {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                 'weight_decay': 0.0}]
            optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon, amsgrad=True)
        else:
            # different functions for getting params in KGPT
            optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon, amsgrad=True)
        # if not args.no_lr_decay:
        #     scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
        #                                                 num_training_steps=t_total)
        # else:
        #     scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,
        #                                                 num_training_steps=1000000)
        if args.scheduler == "linear":
            scheduler = OneCycleLR(optimizer, max_lr=args.learning_rate, total_steps=t_total, pct_start=0.01,
                                   anneal_strategy=args.scheduler, cycle_momentum=True)
        else:
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=2)
        if not args.do_pretrain:
            train(args=args, logger=logger, model=model, train_dataloader=train_dataloader,
                  dev_dataloader=dev_dataloader, optimizer=optimizer, scheduler=scheduler, tokenizer=tokenizer,
                  eval_cls_model=cls_model, eval_cls_tokenizer=cls_tokenizer, eval_cls_start_step=cls_start_step,
                  eval_cls_threshold=cls_threshold, eval_promotion_weight=promotion_weight)
        else:
            pretrain(args, logger, model, train_dataloader, optimizer, scheduler)

    if args.do_predict:
        # Inference on the test set
        if ("kgpt" not in args.model_name) and ("out" in args.output_dir):
            checkpoint = args.output_dir
        else:
            checkpoint = args.model_path

        if "bart-thu" in args.model_name:
            model = MyBart.from_pretrained(checkpoint)
        elif "t5-thu" in args.model_name:
            model = MyT5.from_pretrained(checkpoint)
        elif "bart" in args.model_name:
            model = BartForConditionalGeneration.from_pretrained(checkpoint)
        elif "t5" in args.model_name:
            model = T5ForConditionalGeneration.from_pretrained(checkpoint)
        #  add definition of KGPT models
        elif "seq" in args.model_name:
            model = GatedTransformerDecoder(config, 8, 6)
        elif "graph" in args.model_name:
            model = GraphGatedTransformerDecoder(config, 8, 6)
        else:
            model = TransformerDecoder(config, 8, 6)
        if "bart" in args.model_name or "t5" in args.model_name:
            model.decode_tokenizer = tokenizer
        if "kgpt" in args.model_name or args.n_gpu > 1:
            model = torch.nn.DataParallel(model)
        #  because KGPT are just torch models, not transformer models, so the loading is different
        # and loading should be after DP
        if "kgpt" in args.model_name:
            reloaded = torch.load(checkpoint, map_location=torch.device('cpu'))
            load_my_state_dict(model, reloaded)

        logger.info("Loading checkpoint from {}".format(checkpoint))
        if torch.cuda.is_available():
            model.to(torch.device("cuda"))
        model.eval()
        ems = inference(model=model, dev_dataloader=dev_dataloader, tokenizer=tokenizer, args=args, logger=logger,
                        cls_model=cls_model, cls_tokenizer=cls_tokenizer, save_predictions=True,
                        cls_start_step=cls_start_step, cls_threshold=cls_threshold, promotion_weight=promotion_weight)
        # logger.info("%s on %s data: %.6f" % (dev_dataloader.dataset.metric, dev_dataloader.dataset.data_type, ems))
        # logger.info("%s on data: %.6f" % ("METEOR", ems["METEOR"]))
        logger.info("BERT_R %.6f, METEOR %.6f BLEU4 %.6f ROUGE_L %.6f CIDER %.6f CLS %.6f" %
                    (ems["BERT_R"], ems["METEOR"], ems["Bleu_4"], ems["ROUGE_L"], ems["CIDEr"], ems["CLS"]))


def pretrain(args, logger, model, train_dataloader, optimizer, scheduler):
    model.train()
    global_step = 0
    train_losses = []
    train_enc_loss, train_dec_loss, train_ot_loss = [], [], []
    task_ratio = eval(args.task_ratio)

    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
    logger.info("Starting pretraining!")
    for epoch in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for batch in epoch_iterator:
            global_step += 1
            if torch.cuda.is_available():
                batch = [b.to(torch.device("cuda")) for b in batch]
            task_prob = random.random()
            if global_step == 1:
                for tmp_id in range(28):
                    print(batch[tmp_id])
            # Conduct the three subtasks with the probability in task_ratio
            if task_prob < task_ratio[0]:
                # complete graph + masked text (ar)
                loss = model(input_ids=batch[0], attention_mask=batch[1],
                             decoder_input_ids=batch[2], decoder_attention_mask=batch[3], input_node_ids=batch[4],
                             input_edge_ids=batch[5], node_length=batch[6], edge_length=batch[7], adj_matrix=batch[8],
                             is_training=True)
            else:
                if task_prob < task_ratio[0] + task_ratio[1]:
                    # masked graph + complete text (ae)
                    loss = model(input_ids=batch[9], attention_mask=batch[10], encoder_label=batch[11],
                                 decoder_input_ids=batch[2], decoder_attention_mask=batch[3], input_node_ids=batch[12],
                                 input_edge_ids=batch[13], node_length=batch[14], edge_length=batch[15],
                                 adj_matrix=batch[16], is_training=True)
                else:
                    # complete graph + complete text (ot)
                    loss = model(input_ids=batch[17], attention_mask=batch[18],
                                 decoder_input_ids=batch[19], decoder_attention_mask=batch[20],
                                 decoder_whole_ids=batch[21], input_node_ids=batch[22], input_edge_ids=batch[23],
                                 node_length=batch[24], edge_length=batch[25], word_length=batch[26],
                                 adj_matrix=batch[27], is_training=True)
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.
            if torch.isnan(loss).data:
                logger.info("Stop training because loss=%s" % (loss.data))
                break

            # Record training loss
            train_losses.append(loss.detach().cpu())
            if task_prob < task_ratio[0]:
                train_dec_loss.append(loss.detach().cpu())
            else:
                if task_prob < task_ratio[0] + task_ratio[1]:
                    train_enc_loss.append(loss.detach().cpu())
                else:
                    train_ot_loss.append(loss.detach().cpu())
            loss.backward()

            # Gradient accumulation
            if global_step % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                model.zero_grad()

            # Print loss
            if global_step % args.eval_period == 0:
                enc_loss_res = np.mean(train_enc_loss) if len(train_enc_loss) > 0 else 0.0
                dec_loss_res = np.mean(train_dec_loss) if len(train_dec_loss) > 0 else 0.0
                ot_loss_res = np.mean(train_ot_loss) if len(train_ot_loss) > 0 else 0.0
                logger.info("Step %d Encoder loss %.2f Decoder loss %.2f OT loss %.2f Learning rate %.2e epoch=%d" % (
                    global_step,
                    enc_loss_res,
                    dec_loss_res,
                    ot_loss_res,
                    scheduler.get_last_lr()[0],
                    epoch))
                train_losses = []
                train_enc_loss, train_dec_loss, train_ot_loss = [], [], []

            # Save model
            if global_step % args.save_period == 0:
                if "kgpt" not in args.model_name:
                    model_to_save = model.module if hasattr(model, 'module') else model
                    model_to_save.save_pretrained(args.output_dir)
                else:
                    torch.save(model.state_dict(), '{}/model_ep{}.pt'.format(args.output_dir, epoch))

                logger.info("Saving model on epoch=%d, global_step=%d" % (epoch, global_step))
        if "kgpt" not in args.model_name:
            model_to_save = model.module if hasattr(model, 'module') else model
            model_to_save.save_pretrained(args.output_dir)
            logger.info("Saving model on epoch=%d, global_step=%d" % (epoch, global_step))
        else:
            torch.save(model.state_dict(), '{}/model_ep{}.pt'.format(args.output_dir, epoch))


def train(args, logger, model, train_dataloader, dev_dataloader, optimizer, scheduler, tokenizer, eval_cls_model,
          eval_cls_tokenizer, eval_cls_start_step, eval_cls_threshold, eval_promotion_weight):
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    model.train()
    global_step = 0
    wait_step = 0
    train_losses = []
    best_accuracy = -1
    stop_training = False
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
    logger.info("Starting training!")
    ce_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    for epoch in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for batch in epoch_iterator:
            global_step += 1
            b_input_id = batch[0].to(torch.device("cuda"))
            b_attention_mask = batch[1].to(torch.device("cuda"))
            b_decoder_label = batch[2].to(torch.device("cuda"))
            b_decoder_attention = batch[3].to(torch.device("cuda"))
            b_node_id = batch[4].to(torch.device("cuda"))
            b_edge_id = batch[5].to(torch.device("cuda"))
            b_node_len = batch[6].to(torch.device("cuda"))
            b_edge_len = batch[7].to(torch.device("cuda"))
            b_matrix = batch[8].to(torch.device("cuda"))
            # if torch.cuda.is_available():
            #     batch = [b.to(torch.device("cuda")) for b in batch]
            optimizer.step()
            model.zero_grad()
            optimizer.zero_grad()
            if "t5-thu" in args.model_name:
                logits = model(input_ids=b_input_id, attention_mask=b_attention_mask, decoder_input_ids=b_decoder_label,
                               decoder_attention_mask=b_decoder_attention, input_node_ids=b_node_id,
                               input_edge_ids=b_edge_id, node_length=b_node_len, edge_length=b_edge_len,
                               adj_matrix=b_matrix, is_training=True, return_dict=True, labels=b_decoder_label).logits
                loss = ce_loss_fct(logits.view(-1, logits.size(-1)), b_decoder_label.view(-1))
            elif "bart-thu" in args.model_name:
                decoder_input_ids = b_decoder_label.new_zeros(b_decoder_label.shape)
                decoder_input_ids[:, 1:] = b_decoder_label[:, :-1].clone()
                logits = model(input_ids=b_input_id, attention_mask=b_attention_mask,
                               decoder_input_ids=decoder_input_ids, decoder_attention_mask=b_decoder_attention,
                               input_node_ids=b_node_id, input_edge_ids=b_edge_id, node_length=b_node_len,
                               edge_length=b_edge_len, adj_matrix=b_matrix, is_training=True, return_dict=True,
                               labels=b_decoder_label).logits
                loss = ce_loss_fct(logits.view(-1, logits.size(-1)), b_decoder_label.view(-1))
            elif "t5" in args.model_name:
                #  change output of original t5 to the same format as this code base
                decoder_input_ids = model._shift_right(b_decoder_label)
                logits = model(input_ids=b_input_id, attention_mask=b_attention_mask,
                               decoder_input_ids=decoder_input_ids, decoder_attention_mask=b_decoder_attention,
                               labels=b_decoder_label).logits
                loss = ce_loss_fct(logits.view(-1, logits.size(-1)), b_decoder_label.view(-1))
            elif "bart" in args.model_name:
                #  do shift right outside
                decoder_input_ids = b_decoder_label.new_zeros(b_decoder_label.shape)
                decoder_input_ids[:, 1:] = b_decoder_label[:, :-1].clone()
                logits = model(input_ids=b_input_id, attention_mask=b_attention_mask,
                               decoder_input_ids=decoder_input_ids, decoder_attention_mask=b_decoder_attention,
                               labels=b_decoder_label).logits
                loss = ce_loss_fct(logits.view(-1, logits.size(-1)), b_decoder_label.view(-1))
                #  try to abolish loss below but use labels above
                # ce_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
                # loss = ce_loss_fct(output.view(-1, output.shape[-1]), batch[2].view(-1))
            else:
                #  KGPT output
                output_prob = model(*batch[:-1])
                #  here they don't have loss func inside
                loss = ce_loss_fct(output_prob.view(-1, output_prob.shape[-1]), batch[-1].view(-1))

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.
            if torch.isnan(loss).data:
                logger.info("Stop training because loss=%s" % (loss.data))
                stop_training = True
                break
            train_losses.append(loss.detach().cpu())
            loss.backward()
            # Gradient accumulation
            # if global_step % args.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            # Print loss and evaluate on the valid set
            # if global_step % args.eval_period == 0:
        with torch.no_grad():
            model.eval()
            curr_em = inference(model=model if args.n_gpu == 1 else model.module, dev_dataloader=dev_dataloader,
                                tokenizer=tokenizer, args=args, logger=logger, cls_model=eval_cls_model,
                                cls_tokenizer=eval_cls_tokenizer, cls_start_step=eval_cls_start_step,
                                cls_threshold=eval_cls_threshold, promotion_weight=eval_promotion_weight,
                                save_predictions=False)

            logger.info("Step %d Train loss %.2f Learning rate %.2e %s %.6f on epoch=%d" % (
                global_step,
                np.mean(train_losses),
                scheduler.get_last_lr()[0],
                # dev_dataloader.dataset.metric, #  fix logging error
                "METEOR",
                curr_em["METEOR"],
                epoch))
            train_losses = []
            # fixed bug: do with the initial status = -1, no matter how it should be updated
            if best_accuracy < curr_em["METEOR"] or best_accuracy < 0:
                # Save model
                if "kgpt" not in args.model_name:
                    model_to_save = model.module if hasattr(model, 'module') else model
                    model_to_save.save_pretrained(args.output_dir)
                else:
                    torch.save(model.state_dict(), '{}/pytorch_model.pt'.format(args.output_dir))
                logger.info("Saving model with best %s: %.6f -> %.6f on epoch=%d, global_step=%d" %
                            ("METEOR", best_accuracy, curr_em["METEOR"], epoch, global_step))
                logger.info("BERT_R %.6f METEOR %.6f BLEU4 %.6f ROUGE_L %.6f CIDER %.6f "
                            "CLS %.6f" %
                            (curr_em["BERT_R"], curr_em["METEOR"], curr_em["Bleu_4"],
                             curr_em["ROUGE_L"], curr_em["CIDEr"], curr_em["CLS"]))
                best_accuracy = curr_em["METEOR"]
                wait_step = 0
                stop_training = False
            else:
                wait_step += 1
                if wait_step >= args.wait_step:
                    stop_training = True
                    break
            model.train()
            torch.cuda.empty_cache()
        scheduler.step()
        if stop_training:
            break


def inference(model, dev_dataloader, tokenizer, args, logger, cls_model, cls_tokenizer, cls_start_step, cls_threshold,
              promotion_weight, save_predictions=False):
    #         cls_tokenizer = AutoTokenizer.from_pretrained(args.cls_model)
    #         cls_model = AutoModelForSequenceClassification.from_pretrained(args.cls_model)
    #         cls_start_step = args.cls_start_step
    #         cls_threshold = args.cls_threshold
    predictions = []
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()
    # data_ref = [[x.lower() for x in data_ele['text']] for data_ele in dev_dataloader.dataset.data]
    data_ref = [[x.split("?")[0].strip(" ?.").lower() for x in data_ele['text']]
                for data_ele in dev_dataloader.dataset.data]
    # Inference on the test set
    all_cls_score = []
    for i, batch in tqdm(enumerate(dev_dataloader), "eval process"):
        # batch = [b.to(torch.device("cuda")) for b in batch]
        b_input_id = batch[0].to(torch.device("cuda"))
        b_attention_mask = batch[1].to(torch.device("cuda"))
        b_entity_constraint = batch[9].to(torch.device("cuda"))
        b_cls_prompts = batch[10]
        constrain_list = []
        for each_c in b_entity_constraint[0]:
            constrain_list.append(PhrasalConstraint([x for x in each_c.cpu().numpy().tolist() if x > 0]))
        if "thu" in args.model_name:
            b_node_id = batch[4].to(torch.device("cuda"))
            b_edge_id = batch[5].to(torch.device("cuda"))
            b_node_len = batch[6].to(torch.device("cuda"))
            b_edge_len = batch[7].to(torch.device("cuda"))
            b_matrix = batch[8].to(torch.device("cuda"))
            # output of T5/Bart in PLM
            if promotion_weight == 0:
                outputs = model.generate(input_ids=b_input_id, attention_mask=b_attention_mask,
                                         input_node_ids=b_node_id, input_edge_ids=b_edge_id, node_length=b_node_len,
                                         edge_length=b_edge_len, adj_matrix=b_matrix, num_beams=args.num_beams,
                                         length_penalty=args.length_penalty,
                                         # constraints=constrain_list if len(constrain_list) > 0 else None,
                                         max_length=args.max_output_length, early_stopping=True, do_sample=False, )
            else:
                outputs = generate(self=model, input_ids=b_input_id, attention_mask=b_attention_mask,
                                   input_node_ids=b_node_id, input_edge_ids=b_edge_id, node_length=b_node_len,
                                   edge_length=b_edge_len, adj_matrix=b_matrix, num_beams=args.num_beams,
                                   length_penalty=args.length_penalty, max_length=args.max_output_length,
                                   early_stopping=True, do_sample=False, cls_prompt_list=b_cls_prompts,
                                   decode_tokenizer=tokenizer, cls_tokenizer=cls_tokenizer, cls_threshold=cls_threshold,
                                   cls_model=cls_model, cls_start_step=cls_start_step,
                                   cls_promotion_weight=promotion_weight)
        elif "kgpt" in args.model_name:
            # specify beams for KGPT
            batch = [b.to(torch.device("cuda")) for b in batch]
            if args.num_beams == 1:
                # index might needs to be changed,
                outputs = model.module.greedy_decode(*batch[:-2], banwords=tokenizer.convert_tokens_to_ids([]),
                                                     max_token_seq_len=args.max_output_length, )
            else:
                outputs = model.module.beam_search(batch[:-2], tokenizer, n_bm=args.num_beams,
                                                   banwords=tokenizer.convert_tokens_to_ids([]),
                                                   max_token_seq_len=args.max_output_length, )
        else:
            # output of T5/Bart in PLM
            if promotion_weight == 0:
                outputs = model.generate(input_ids=b_input_id, attention_mask=b_attention_mask,
                                         num_beams=args.num_beams, length_penalty=args.length_penalty,
                                         # constraints=constrain_list if len(constrain_list) > 0 else None,
                                         max_length=args.max_output_length, early_stopping=True, do_sample=False)
            else:
                outputs = generate(self=model, input_ids=b_input_id, attention_mask=b_attention_mask,
                                   num_beams=args.num_beams, length_penalty=args.length_penalty,
                                   # constraints=constrain_list if len(constrain_list) > 0 else None,
                                   max_length=args.max_output_length, early_stopping=True, do_sample=False,
                                   cls_prompt_list=b_cls_prompts, decode_tokenizer=tokenizer,
                                   cls_tokenizer=cls_tokenizer, cls_threshold=cls_threshold, cls_model=cls_model,
                                   cls_start_step=cls_start_step, cls_promotion_weight=promotion_weight)
        # Convert ids to tokens
        ref_id = 0
        for input_, output in zip(batch[0], outputs):
            # trunk content after eos and before bos, especially bart v4 starts with eos
            if tokenizer.bos_token_id is not None and tokenizer.bos_token_id in output:
                start_idx = (output == tokenizer.bos_token_id).nonzero(as_tuple=True)[0][0]
                output = output[start_idx:]
            if tokenizer.eos_token_id in output:
                stop_idx = (output == tokenizer.eos_token_id).nonzero(as_tuple=True)[0][0] + 1
                output = output[: stop_idx]
            pred = tokenizer.decode(
                output, skip_special_tokens=True, clean_up_tokenization_spaces=args.clean_up_spaces)
            if '[EOS]' in pred:
                pred = pred[: pred.index('[EOS]')]
            if cls_model != "":
                cls_input = [each_cls_prompt[0] + pred.strip(".? ") for each_cls_prompt in b_cls_prompts]
                tokenized_cls = cls_tokenizer(cls_input, truncation=True)
                tokenized_cls["input_ids"] = pad_sequence(
                    [torch.LongTensor(x) for x in tokenized_cls["input_ids"]],
                    padding_value=cls_tokenizer.pad_token_id, batch_first=True).to(torch.device("cuda"))
                tokenized_cls["attention_mask"] = pad_sequence(
                    [torch.LongTensor(x) for x in tokenized_cls["attention_mask"]], padding_value=0,
                    batch_first=True).to(torch.device("cuda"))
                cls_output = cls_model(input_ids=tokenized_cls["input_ids"], attention_mask=tokenized_cls["attention_mask"])
                softmax_cls = torch.softmax(cls_output.logits, dim=1, )
                all_cls_score.extend([float(x[1]) for x in softmax_cls])
                # print([float(x[1]) for x in softmax_cls])
                # predictions.append(pred)
                # if len(b_cls_prompts) > 1:
                #     print(b_cls_prompts)
                #     print(pred)
                # print(len(output), len(pred.split()))
            predictions.append(pred.split("?")[0].strip(" .").lower())
            ref_id += 1
    # Save the generated results
    if save_predictions:
        save_path = os.path.join(args.output_dir, "{}predictions.txt".format(args.prefix))
        with open(save_path, "w") as f:
            for pred in predictions:
                f.write(pred + '\n')
        logger.info("Saved prediction in {}".format(save_path))
    assert len(predictions) == len(data_ref)
    torch.cuda.empty_cache()
    tmp_ref = [x[0] for x in data_ref]
    bert_p, bert_r, bert_f1 = score(predictions, tmp_ref, model_type="roberta-large", verbose=True)
    # tmp_ref = data_ref
    metrics = {"BERT_R": np.mean(bert_r.numpy()),
               "CLS": np.mean(all_cls_score)}
    metrics.update(evaluate_bleu(data_ref=data_ref, data_sys=predictions))
    return metrics
