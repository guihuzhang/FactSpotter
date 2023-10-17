#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python train_classifier_grail.py \
  --model_name "google/electra-small-discriminator" \
  --learning_rate 5e-5 \
  --b_size 64 \
  --workers 4 \
  --epochs 15 \
  --output_prefix electra_small_grail.5pred.25ent.25both.1drop.01aug \
  --append_test False \
  --drop_percent 0.1 \
  --neg_pred_percent 0.5

CUDA_VISIBLE_DEVICES=0 python train_classifier_grail.py \
  --model_name "google/electra-small-discriminator" \
  --learning_rate 5e-5 \
  --b_size 64 \
  --workers 4 \
  --epochs 15 \
  --output_prefix electra_grail_no_test.4pred.3ent.3both.1drop \
  --append_test False \
  --drop_percent 0.1 \
  --neg_pred_percent 0.4
