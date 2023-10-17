#!/bin/bash

#SBATCH --job-name=test_webnlg_const_bart_base
#SBATCH -C v100-32g
#SBATCH -A vtv@v100
##SBATCH --partition=gpu_p2          # uncomment for gpu_p2 partition gpu_p2
#SBATCH --nodes=1                    # we request one node
#SBATCH --ntasks-per-node=1          # with one task per node (= number of GPUs here)
#SBATCH --gres=gpu:1                 # number of GPUs (1/4 of GPUs)
#SBATCH --cpus-per-task=10           # number of cores per task (1/4 of the 4-GPUs node)
##SBATCH --cpus-per-task=3           # number of cores per task (with gpu_p2: 1/8 of the 8-GPUs node)
## /!\ Caution, "multithread" in Slurm vocabulary refers to hyperthreading.
#SBATCH --hint=multithread           # hyperthreading is activated
##SBATCH --qos=qos_gpu-t4
#SBATCH --time=20:00:00             # maximum execution time requested (HH:MM:SS)
#SBATCH --output=out/log/test_webnlg_const_bart_base%j.out    # name of output file
#SBATCH --error=out/log/test_webnlg_const_bart_base%j.out     # name of error file (here, in common with the output file)

module purge

# use new stable pytorch for compatibility
module load pytorch-gpu/py3/1.13.0
module load openjdk/1.8.0_222-b10

set -x

##cat /proc/cpuinfo
nvidia-smi

CUDA_VISIBLE_DEVICES=0 TRANSFORMERS_OFFLINE=1 python code/cli_gt.py \
        --do_predict \
        --model_name bart-base \
        --output_dir out/webnlg_const_bart_base_full \
        --train_file ../data/webnlg_const_new/train \
        --predict_file ../data/webnlg_const_new/test \
        --tokenizer_path facebook/bart-base \
        --dataset webnlg_const_new \
        --append_another_bos \
        --predict_batch_size 1 \
        --max_input_length 256 \
        --max_output_length 256 \
        --num_beams 5 \
        --length_penalty 1. \
        --prefix webnlg_5beam.1weight_ \
        --cls_model "google/electra-small-discriminator" \
        --cls_model_path "electra_webnlg_fix.5pred.25ent.25both.1drop0.9888319088319089.pt" \
        --promotion_weight .1

CUDA_VISIBLE_DEVICES=0 TRANSFORMERS_OFFLINE=1 python code/cli_gt.py \
        --do_predict \
        --model_name bart-base \
        --output_dir out/webnlg_const_bart_base_full \
        --train_file ../data/webnlg_const_new/train \
        --predict_file ../data/webnlg_const_new/test \
        --tokenizer_path facebook/bart-base \
        --dataset webnlg_const_new \
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
