#!/bin/bash

CUDA_VISIBLE_DEVICES=0 TRANSFORMERS_OFFLINE=1 python code/cli_gt.py \
        --do_predict \
        --model_name t5-thu \
        --output_dir out/grail_new_t5jointgt \
        --train_file ../data/grail_new/train \
        --predict_file ../data/grail_new/iid \
        --tokenizer_path pretrain_model/jointgt_t5 \
        --dataset grail_new \
        --predict_batch_size 1 \
        --append_another_bos \
        --max_input_length 512 \
        --max_output_length 256 \
        --num_beams 5 \
        --length_penalty 1. \
        --prefix grail_iid_t5joint_beam5weight.1 \
        --cls_model "google/electra-small-discriminator" \
        --cls_model_path "electra_grail_fix.5pred.25ent.25both.1drop0.9670569557451876.pt" \
        --promotion_weight .1

CUDA_VISIBLE_DEVICES=0 TRANSFORMERS_OFFLINE=1 python code/cli_gt.py \
        --do_predict \
        --model_name t5-thu \
        --output_dir out/grail_new_t5jointgt \
        --train_file ../data/grail_new/train \
        --predict_file ../data/grail_new/iid \
        --tokenizer_path pretrain_model/jointgt_t5 \
        --dataset grail_new \
        --predict_batch_size 1 \
        --append_another_bos \
        --max_input_length 512 \
        --max_output_length 256 \
        --num_beams 5 \
        --length_penalty 1. \
        --prefix grail_iid_t5joint_beam5weight.15 \
        --cls_model "google/electra-small-discriminator" \
        --cls_model_path "electra_grail_fix.5pred.25ent.25both.1drop0.9670569557451876.pt" \
        --promotion_weight .15