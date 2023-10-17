#!/bin/bash

CUDA_VISIBLE_DEVICES=0 TRANSFORMERS_OFFLINE=1 python code/cli_gt.py \
        --do_train \
        --model_name bart-large \
        --output_dir out/spq_predicate_zero_bart_large \
        --train_file ../data/spq_predicate_zero/train \
        --predict_file ../data/spq_predicate_zero/val \
        --model_path facebook/bart-large \
        --tokenizer_path facebook/bart-large \
        --dataset spq_predicate_zero \
        --train_batch_size 18  \
        --predict_batch_size 1 \
        --max_input_length 512 \
        --max_output_length 64 \
        --append_another_bos \
        --learning_rate 5e-5 \
        --num_train_epochs 64 \
        --wait_step 32 \
        --num_beams 2 \
        --num_workers 8 \
        --scheduler cosine \
        --cls_model "google/electra-small-discriminator" \
        --cls_model_path "electra_spq.5pred.25ent.25both.1drop0.9551447538262954.pt" \
        --promotion_weight 0


CUDA_VISIBLE_DEVICES=0 TRANSFORMERS_OFFLINE=1 python code/cli_gt.py \
        --do_predict \
        --model_name bart-large \
        --output_dir out/spq_predicate_zero_bart_large \
        --train_file ../data/spq_predicate_zero/train \
        --predict_file ../data/spq_predicate_zero/test \
        --tokenizer_path facebook/bart-large \
        --dataset spq_predicate_zero \
        --append_another_bos \
        --predict_batch_size 1 \
        --max_input_length 512 \
        --max_output_length 64 \
        --num_beams 5 \
        --length_penalty 1. \
        --prefix spq_predicate_zero_baseline_bart_large_ \
        --cls_model "google/electra-small-discriminator" \
        --cls_model_path "electra_spq.5pred.25ent.25both.1drop0.9551447538262954.pt" \
        --promotion_weight 0
