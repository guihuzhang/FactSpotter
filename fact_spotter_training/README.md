# Training Code of FactSpotter

This is codebase of training FactSpotter metric. 

To train it on different dataset, please run the following scripts:
   
    bash train_classifier_grail.py
    bash train_classifier_spq.py
    bash train_classifier_dart.py
    bash train_classifier_webnlg.py

The test sets are fixed. Run following scripts to get accuracy, 
F1, confusion matrix of FactSpotter on test split:

    bash test_classifier_json.py 

You can change variables `model_state_dict` and `eval_dataset` 
to get performance of different datasets by different models.