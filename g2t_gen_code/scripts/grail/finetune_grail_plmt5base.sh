#!/bin/bash

CUDA_VISIBLE_DEVICES=0 TRANSFORMERS_OFFLINE=1 python code/cli_gt.py \
        --do_train \
        --model_name t5-base \
        --output_dir out/grail_new_t5base \
        --train_file ../data/grail_new/train \
        --predict_file ../data/grail_new/val \
        --model_path t5-base \
        --tokenizer_path t5-base \
        --dataset grail_new \
        --train_batch_size 16 \
        --predict_batch_size 1 \
        --max_input_length 512 \
        --max_output_length 256 \
        --append_another_bos \
        --learning_rate 1e-4 \
        --num_train_epochs 64 \
        --wait_step 32 \
        --num_beams 2 \
        --num_workers 8 \
        --length_penalty 1. \
        --scheduler cosine \
        --cls_model "google/electra-small-discriminator" \
        --cls_model_path "electra_grail_fix.5pred.25ent.25both.1drop0.9670569557451876.pt" \
        --promotion_weight 0

CUDA_VISIBLE_DEVICES=0 TRANSFORMERS_OFFLINE=1 python code/cli_gt.py \
        --do_predict \
        --model_name t5-base \
        --output_dir out/grail_new_t5base \
        --train_file ../data/grail_new/train \
        --predict_file ../data/grail_new/test \
        --tokenizer_path t5-base \
        --dataset grail_new \
        --append_another_bos \
        --predict_batch_size 1 \
        --max_input_length 512 \
        --max_output_length 256 \
        --num_beams 5 \
        --length_penalty 1. \
        --prefix grail_test_baseline_t5base_ \
        --cls_model "google/electra-small-discriminator" \
        --cls_model_path "electra_grail_fix.5pred.25ent.25both.1drop0.9670569557451876.pt" \
        --promotion_weight 0

CUDA_VISIBLE_DEVICES=0 TRANSFORMERS_OFFLINE=1 python code/cli_gt.py \
        --do_predict \
        --model_name t5-base \
        --output_dir out/grail_new_t5base \
        --train_file ../data/grail_new/train \
        --predict_file ../data/grail_new/iid \
        --tokenizer_path t5-base \
        --dataset grail_new \
        --append_another_bos \
        --predict_batch_size 1 \
        --max_input_length 512 \
        --max_output_length 256 \
        --num_beams 5 \
        --length_penalty 1. \
        --prefix grail_iid_baseline_t5base_ \
        --cls_model "google/electra-small-discriminator" \
        --cls_model_path "electra_grail_fix.5pred.25ent.25both.1drop0.9670569557451876.pt" \
        --promotion_weight 0

CUDA_VISIBLE_DEVICES=0 TRANSFORMERS_OFFLINE=1 python code/cli_gt.py \
        --do_predict \
        --model_name t5-base \
        --output_dir out/grail_new_t5base \
        --train_file ../data/grail_new/train \
        --predict_file ../data/grail_new/compositional \
        --tokenizer_path t5-base \
        --dataset grail_new \
        --predict_batch_size 1 \
        --append_another_bos \
        --max_input_length 512 \
        --max_output_length 256 \
        --num_beams 5 \
        --length_penalty 1. \
        --prefix grail_cmp_baseline_t5base_ \
        --cls_model "google/electra-small-discriminator" \
        --cls_model_path "electra_grail_fix.5pred.25ent.25both.1drop0.9670569557451876.pt" \
        --promotion_weight 0

CUDA_VISIBLE_DEVICES=0 TRANSFORMERS_OFFLINE=1 python code/cli_gt.py \
        --do_predict \
        --model_name t5-base \
        --output_dir out/grail_new_t5base \
        --train_file ../data/grail_new/train \
        --predict_file ../data/grail_new/zero-shot \
        --tokenizer_path t5-base \
        --dataset grail_new \
        --predict_batch_size 1 \
        --append_another_bos \
        --max_input_length 512 \
        --max_output_length 256 \
        --num_beams 5 \
        --length_penalty 1. \
        --prefix grail_zero_baseline_t5base_ \
        --cls_model "google/electra-small-discriminator" \
        --cls_model_path "electra_grail_fix.5pred.25ent.25both.1drop0.9670569557451876.pt" \
        --promotion_weight 0
