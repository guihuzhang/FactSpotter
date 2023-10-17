# FactSpotter

Official PyTorch implementation for EMNLP 2023 Finding paper, "FactSpotter: Evaluating the Factual Faithfulness of Graph-to-Text Generation"

This repo contains four folders:

`fact_spotter_training` is the training code for our proposed metric, FactSpotter;

`g2t_gen_code` is the training code for baseline models of T5, BART, and JointGT for G2T generation, and 
FactSpotter is integrated in inference code to promote factual G2T generation. 

`webnlg-human-judgement` computes the correlation of metrics to human evaluation.

`data`, should contain JSON format files of G2T generation datasets same as [JointGT](https://github.com/thu-coai/JointGT).  

In each folder there is a `README.md` which tells how to run their codes.

File `generation_sample_annotation_v2` has human annotation for generated samples 
before and after using FactSpotter in generation.
