#!/bin/bash

rec_fpath=../samples/1bcu/1bcu_protein.pdb
lig_fpath=../samples/1bcu/1bcu_ligand.pdb
out_fpath=1bcu_pred-pKd.csv

scaler_fpath=../models/train_scaler.scaler
model_fpath=../models/62shell_saved-model.h5

python predict.py \
    -rec_fpath $rec_fpath \
    -lig_fpath $lig_fpath \
    -shape 84,124,1 \
    -scaler $scaler_fpath \
    -model $model_fpath \
    -shells 62 \
    -out_fpath $out_fpath