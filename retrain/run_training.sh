#!/bin/bash
# Here is the code for training OnionNet-2 when the total number of shells is 62.


train_fpath=final_valid_features_true.pkl  # Specify the path to your training set
valid_fpath=final_valid_features_true.pkl  # Specify the path to your validating set
label_fpath=PDBbindv2019_pKd-label.csv

python train.py \
    -train_fpath $train_fpath \
    -valid_fpath $valid_fpath \
    -label_fpath $label_fpath \
    -shape 84,124,1 \
    -batch_size 64 \
    -rate 0.0 \
    -alpha 0.7 \
    -clipvalue 0.01 \
    -n_features 10416 \
    -epochs 300 \
    -out_model 62shell_saved-model.h5