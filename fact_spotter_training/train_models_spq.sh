#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python train_classifier_spq.py \
  --model_name "google/electra-small-discriminator" \
  --learning_rate 5e-5 \
  --b_size 1 \
  --workers 1 \
  --epochs 15 \
  --output_prefix electra_grail.5pred.25ent.25both.1drop \
  --append_test False \
  --drop_percent 0.1 \
  --neg_pred_percent 0.5

