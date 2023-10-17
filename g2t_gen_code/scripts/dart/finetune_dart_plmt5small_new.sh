#!/bin/bash

CUDA_VISIBLE_DEVICES=0 TRANSFORMERS_OFFLINE=1 python code/cli_gt.py \
        --do_train \
        --model_name t5-small \
        --output_dir out/dart_t5small_new \
        --train_file ../data/dart_new/train_new \
        --predict_file ../data/dart_new/val_new \
        --model_path t5-small \
        --tokenizer_path t5-small \
        --dataset dart_new \
        --train_batch_size 32 \
        --predict_batch_size 1 \
        --max_input_length 512 \
        --max_output_length 256 \
        --append_another_bos \
        --learning_rate 1e-4 \
        --num_train_epochs 32 \
        --wait_step 32 \
        --num_beams 2 \
        --num_workers 8 \
        --scheduler cosine \
        --cls_model "google/electra-small-discriminator" \
        --cls_model_path "electra_dart.5pred.25ent.25both.1drop0.9921646746347942.pt" \
        --promotion_weight .0

CUDA_VISIBLE_DEVICES=0 TRANSFORMERS_OFFLINE=1 python code/cli_gt.py \
        --do_predict \
        --model_name t5-small \
        --output_dir out/dart_t5small_new \
        --train_file ../data/dart_new/train_new \
        --predict_file ../data/dart_new/test_new \
        --tokenizer_path t5-small \
        --dataset dart_new \
        --append_another_bos \
        --predict_batch_size 1 \
        --max_input_length 512 \
        --max_output_length 256 \
        --num_beams 5 \
        --length_penalty 1. \
        --prefix dart_new_baseline_t5s_ \
        --cls_model "google/electra-small-discriminator" \
        --cls_model_path "electra_dart.5pred.25ent.25both.1drop0.9921646746347942.pt" \
        --promotion_weight .0

CUDA_VISIBLE_DEVICES=0 TRANSFORMERS_OFFLINE=1 python code/cli_gt.py \
        --do_predict \
        --model_name t5-small \
        --output_dir out/dart_t5small_new \
        --train_file ../data/dart_new/train_new \
        --predict_file ../data/dart_new/test_new \
        --tokenizer_path t5-small \
        --dataset dart_new \
        --append_another_bos \
        --predict_batch_size 1 \
        --max_input_length 512 \
        --max_output_length 256 \
        --num_beams 5 \
        --length_penalty 1. \
        --prefix dart_new_t5s.15 \
        --cls_model "google/electra-small-discriminator" \
        --cls_model_path "electra_dart.5pred.25ent.25both.1drop0.9921646746347942.pt" \
        --promotion_weight .15

CUDA_VISIBLE_DEVICES=0 TRANSFORMERS_OFFLINE=1 python code/cli_gt.py \
        --do_predict \
        --model_name t5-small \
        --output_dir out/dart_t5small_new \
        --train_file ../data/dart_new/train_new \
        --predict_file ../data/dart_new/test_new \
        --tokenizer_path t5-small \
        --dataset dart_new \
        --append_another_bos \
        --predict_batch_size 1 \
        --max_input_length 512 \
        --max_output_length 256 \
        --num_beams 5 \
        --length_penalty 1. \
        --prefix dart_new_t5s.1 \
        --cls_model "google/electra-small-discriminator" \
        --cls_model_path "electra_dart.5pred.25ent.25both.1drop0.9921646746347942.pt" \
        --promotion_weight .1