
## G2T Generation Codes

This folder is a PyTorch implementation of several graph-to-text (G2T) generation networks on HuggingFace Transformers 
4, partially based on previous works [KGPT](https://github.com/wenhuchen/KGPT), 
[PLM](https://github.com/UKPLab/plms-graph2text), and [JointGT](https://github.com/thu-coai/JointGT).

We include models of T5, BART, and JointGT for G2T generation, and we integrated FactSpotter into the inference to 
improve the factual faithfulness of generation.


## Dependencies

* Python 3.8+
* PyTorch 1.13
* Transformers (Huggingface) 4
* PyTorch Scatter 2.0.9

## Datasets

Our experiments contain several datasets, i.e., WebNLG, DART, SimpleQuestions and GrailQA.
They've been pre-processed to JSON format and put under the directory data/.

## Fine-tuning

T5 and BART model are able to be automatically loaded from HuggingFace library. For fine-tuning on JointGT, please
firstly download their checkpoint of pre-trained model 
([Google Drive](https://drive.google.com/drive/folders/1FGThWaTUs1cLvkd_GHCFV8mQEDW6qfIK?usp=sharing) /
[Tsinghua Cloud](https://cloud.tsinghua.edu.cn/d/79b009058cce484fa736/)), 
and fine-tune the pre-trained model on downstream datasets.

To train a certain model on a dataset, please execute following command:
```shell
bash scripts/finetune_grail_plmt5base.sh
```
`--output_dir` is the directory to save the fine-tuning model. 

`--model_path` is the pre-trained checkpoint used for fine-tuning. 

`--cls_model` is the type of model used by FactSpotter

`--cls_model_path` is directory of trained FactSpotter weight.

`--promotion_weight` is the weight for promoting factual generation. 
When `promotion_weight=0` it executes original beam search, 

## Thanks

Many thanks to the GitHub repositories of [Transformers](https://github.com/huggingface/transformers), [bart-closed-book-qa](https://github.com/shmsw25/bart-closed-book-qa) [KGPT](https://github.com/wenhuchen/KGPT), [PLM](https://github.com/UKPLab/plms-graph2text), and [JointGT](https://github.com/thu-coai/JointGT). 
Part of our codes are modified based on their codes.
