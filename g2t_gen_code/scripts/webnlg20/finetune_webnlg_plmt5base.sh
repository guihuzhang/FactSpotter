#!/bin/bash

CUDA_VISIBLE_DEVICES=0 TRANSFORMERS_OFFLINE=1 python code/cli_gt.py \
        --do_train \
        --model_name t5-base \
        --output_dir out/webnlg20t5base_full \
        --train_file data/webnlg20/train \
        --predict_file data/webnlg20/val \
        --model_path t5-base \
        --tokenizer_path t5-base \
        --dataset webnlg20 \
        --train_batch_size 28 \
        --predict_batch_size 1 \
        --max_input_length 256 \
        --max_output_length 256 \
        --append_another_bos \
        --learning_rate 1e-4 \
        --num_train_epochs 64 \
        --wait_step 32 \
        --num_beams 2 \
        --num_workers 8 \
        --scheduler cosine \
        --cls_model "google/electra-small-discriminator" \
        --cls_model_path "electra_small_webnlg20.5pred.25ent.25both.1drop.1reverse.01augment0.9862945289091111.pt" \
        --promotion_weight .0

CUDA_VISIBLE_DEVICES=0 TRANSFORMERS_OFFLINE=1 python code/cli_gt.py \
        --do_predict \
        --model_name t5-base \
        --output_dir out/webnlg20t5base_full \
        --train_file data/webnlg20/train \
        --predict_file data/webnlg20/test \
        --tokenizer_path t5-base \
        --dataset webnlg20 \
        --append_another_bos \
        --predict_batch_size 1 \
        --max_input_length 256 \
        --max_output_length 256 \
        --num_beams 5 \
        --length_penalty 1. \
        --prefix webnlg_baseline_ \
        --cls_model "google/electra-small-discriminator" \
        --cls_model_path "electra_small_webnlg20.5pred.25ent.25both.1drop.1reverse.01augment0.9862945289091111.pt" \
        --promotion_weight .0
