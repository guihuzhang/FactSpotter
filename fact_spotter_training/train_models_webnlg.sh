#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python train_classifier_webnlg.py \
  --model_name "google/electra-small-discriminator" \
  --learning_rate 5e-5 \
  --dataset_name webnlg \
  --b_size 64 \
  --workers 4 \
  --epochs 15 \
  --output_prefix electra_small_webnlg_antonym.5pred.25ent.25both.1drop.01augment \
  --append_test False \
  --drop_percent 0.1 \
  --neg_pred_percent 0.5
