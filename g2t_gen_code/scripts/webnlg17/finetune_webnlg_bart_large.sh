#!/bin/bash

CUDA_VISIBLE_DEVICES=0 TRANSFORMERS_OFFLINE=1 python code/cli_gt.py \
        --do_train \
        --model_name bart-large \
        --output_dir out/webnlg_bart_large_full \
        --train_file ../data/webnlg_new/train \
        --predict_file ../data/webnlg_new/val \
        --model_path facebook/bart-large \
        --tokenizer_path facebook/bart-large \
        --dataset webnlg_new \
        --train_batch_size 20  \
        --predict_batch_size 1 \
        --max_input_length 256 \
        --max_output_length 256 \
        --append_another_bos \
        --learning_rate 5e-5 \
        --num_train_epochs 64 \
        --num_beams 2 \
        --wait_step 32 \
        --num_workers 8 \
        --scheduler cosine \
        --cls_model "google/electra-small-discriminator" \
        --cls_model_path "electra_webnlg_fix.5pred.25ent.25both.1drop0.9888319088319089.pt" \
        --promotion_weight .0

CUDA_VISIBLE_DEVICES=0 TRANSFORMERS_OFFLINE=1 python code/cli_gt.py \
        --do_predict \
        --model_name bart-large \
        --output_dir out/webnlg_bart_large_full \
        --train_file ../data/webnlg_new/train \
        --predict_file ../data/webnlg_new/test \
        --tokenizer_path facebook/bart-large \
        --dataset webnlg_new \
        --append_another_bos \
        --predict_batch_size 1 \
        --max_input_length 256 \
        --max_output_length 256 \
        --num_beams 5 \
        --length_penalty 1. \
        --prefix webnlg_baseline_bart_large_ \
        --cls_model "google/electra-small-discriminator" \
        --cls_model_path "electra_webnlg_fix.5pred.25ent.25both.1drop0.9888319088319089.pt" \
        --promotion_weight .0
