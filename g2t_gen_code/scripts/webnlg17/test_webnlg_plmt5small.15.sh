#!/bin/bash

CUDA_VISIBLE_DEVICES=0 TRANSFORMERS_OFFLINE=1 python code/cli_gt.py \
        --do_predict \
        --model_name t5-small \
        --output_dir out/webnlg_t5small_full \
        --train_file ../data/webnlg_new/train \
        --predict_file ../data/webnlg_new/test \
        --tokenizer_path t5-small \
        --dataset webnlg_new \
        --append_another_bos \
        --predict_batch_size 1 \
        --max_input_length 256 \
        --max_output_length 256 \
        --num_beams 5 \
        --length_penalty 1. \
        --prefix webnlg_5beam.15weight_ \
        --cls_model "google/electra-small-discriminator" \
        --cls_model_path "electra_webnlg_fix.5pred.25ent.25both.1drop0.9888319088319089.pt" \
        --promotion_weight .15

CUDA_VISIBLE_DEVICES=0 TRANSFORMERS_OFFLINE=1 python code/cli_gt.py \
        --do_predict \
        --model_name t5-small \
        --output_dir out/webnlg_const_t5small_full \
        --train_file ../data/webnlg_const_new/train \
        --predict_file ../data/webnlg_const_new/test \
        --tokenizer_path t5-small \
        --dataset webnlg_const_new \
        --append_another_bos \
        --predict_batch_size 1 \
        --max_input_length 256 \
        --max_output_length 256 \
        --num_beams 5 \
        --length_penalty 1. \
        --prefix webnlg_const5beam.15weight_ \
        --cls_model "google/electra-small-discriminator" \
        --cls_model_path "electra_webnlg_fix.5pred.25ent.25both.1drop0.9888319088319089.pt" \
        --promotion_weight .15