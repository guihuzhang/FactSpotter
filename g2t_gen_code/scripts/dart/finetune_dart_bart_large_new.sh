#!/bin/bash

CUDA_VISIBLE_DEVICES=0 TRANSFORMERS_OFFLINE=1 python code/cli_gt.py \
        --do_train \
        --model_name bart-large \
        --output_dir out/dart_bart_large_new \
        --train_file ../data/dart_new/train_new \
        --predict_file ../data/dart_new/val_new \
        --model_path facebook/bart-large \
        --tokenizer_path facebook/bart-large \
        --dataset dart_new \
        --train_batch_size 12  \
        --predict_batch_size 1 \
        --max_input_length 512 \
        --max_output_length 256 \
        --append_another_bos \
        --learning_rate 5e-5 \
        --num_train_epochs 64 \
        --wait_step 32 \
        --num_beams 2 \
        --num_workers 8 \
        --scheduler cosine \
        --cls_model "google/electra-small-discriminator" \
        --cls_model_path "electra_dart.5pred.25ent.25both.1drop0.9921646746347942.pt" \
        --promotion_weight .0

CUDA_VISIBLE_DEVICES=0 TRANSFORMERS_OFFLINE=1 python code/cli_gt.py \
        --do_predict \
        --model_name bart-large \
        --output_dir out/dart_bart_large_new \
        --train_file ../data/dart_new/train_new \
        --predict_file ../data/dart_new/test_new \
        --tokenizer_path facebook/bart-large \
        --dataset dart_new \
        --append_another_bos \
        --predict_batch_size 1 \
        --max_input_length 512 \
        --max_output_length 256 \
        --num_beams 5 \
        --length_penalty 1. \
        --prefix dart_baseline_bart_large_new_ \
        --cls_model "google/electra-small-discriminator" \
        --cls_model_path "electra_dart.5pred.25ent.25both.1drop0.9921646746347942.pt" \
        --promotion_weight .0

CUDA_VISIBLE_DEVICES=0 TRANSFORMERS_OFFLINE=1 python code/cli_gt.py \
        --do_predict \
        --model_name bart-large \
        --output_dir out/dart_bart_large_new \
        --train_file ../data/dart_new/train_new \
        --predict_file ../data/dart_new/test_new \
        --tokenizer_path facebook/bart-large \
        --dataset dart_new \
        --append_another_bos \
        --predict_batch_size 1 \
        --max_input_length 512 \
        --max_output_length 256 \
        --num_beams 5 \
        --length_penalty 1. \
        --prefix dart_bart_large_new.15 \
        --cls_model "google/electra-small-discriminator" \
        --cls_model_path "electra_dart.5pred.25ent.25both.1drop0.9921646746347942.pt" \
        --promotion_weight .15
        
CUDA_VISIBLE_DEVICES=0 TRANSFORMERS_OFFLINE=1 python code/cli_gt.py \
        --do_predict \
        --model_name bart-large \
        --output_dir out/dart_bart_large_new \
        --train_file ../data/dart_new/train_new \
        --predict_file ../data/dart_new/test_new \
        --tokenizer_path facebook/bart-large \
        --dataset dart_new \
        --append_another_bos \
        --predict_batch_size 1 \
        --max_input_length 512 \
        --max_output_length 256 \
        --num_beams 5 \
        --length_penalty 1. \
        --prefix dart_bart_large_new.1 \
        --cls_model "google/electra-small-discriminator" \
        --cls_model_path "electra_dart.5pred.25ent.25both.1drop0.9921646746347942.pt" \
        --promotion_weight .1
