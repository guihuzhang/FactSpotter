#!/bin/bash

CUDA_VISIBLE_DEVICES=0 TRANSFORMERS_OFFLINE=1 python code/cli_gt.py \
        --do_predict \
        --model_name bart-base \
        --output_dir out/spq_predicate_zero_bart_base \
        --train_file ../data/spq_predicate_zero/train \
        --predict_file ../data/spq_predicate_zero/test \
        --tokenizer_path facebook/bart-base \
        --dataset spq_predicate_zero \
        --append_another_bos \
        --predict_batch_size 1 \
        --max_input_length 512 \
        --max_output_length 64 \
        --num_beams 5 \
        --length_penalty 1. \
        --prefix spq_predicate_zero_bart_base_beam5cls.1 \
        --cls_model "google/electra-small-discriminator" \
        --cls_model_path "electra_spq.5pred.25ent.25both.1drop0.9551447538262954.pt" \
        --promotion_weight .1

CUDA_VISIBLE_DEVICES=0 TRANSFORMERS_OFFLINE=1 python code/cli_gt.py \
        --do_predict \
        --model_name bart-base \
        --output_dir out/spq_predicate_zero_bart_base \
        --train_file ../data/spq_predicate_zero/train \
        --predict_file ../data/spq_predicate_zero/test \
        --tokenizer_path facebook/bart-base \
        --dataset spq_predicate_zero \
        --append_another_bos \
        --predict_batch_size 1 \
        --max_input_length 512 \
        --max_output_length 64 \
        --num_beams 5 \
        --length_penalty 1. \
        --prefix spq_predicate_zero_bart_base_beam5cls.15 \
        --cls_model "google/electra-small-discriminator" \
        --cls_model_path "electra_spq.5pred.25ent.25both.1drop0.9551447538262954.pt" \
        --promotion_weight .15