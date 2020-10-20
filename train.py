#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
import pandas as pd
import numpy as np
from argparse import RawDescriptionHelpFormatter
import argparse
from sklearn import preprocessing
from sklearn.externals import joblib
import os

def PCC_RMSE(y_true, y_pred):
    alpha = args.alpha
    
    fsp = y_pred - tf.keras.backend.mean(y_pred)
    fst = y_true - tf.keras.backend.mean(y_true)

    devP = tf.keras.backend.std(y_pred)
    devT = tf.keras.backend.std(y_true)

    rmse = tf.keras.backend.sqrt(tf.keras.backend.mean(tf.keras.backend.square(y_pred - y_true), axis=-1))

    pcc = 1.0 - tf.keras.backend.mean(fsp * fst) / (devP * devT)
    loss = alpha * pcc + (1 - alpha) * rmse
    
    return loss

def RMSE(y_true, y_pred):
    return tf.keras.backend.sqrt(tf.keras.backend.mean(tf.keras.backend.square(y_pred - y_true), axis=-1))

def PCC(y_true, y_pred):
    fsp = y_pred - tf.keras.backend.mean(y_pred)
    fst = y_true - tf.keras.backend.mean(y_true)

    devP = tf.keras.backend.std(y_pred)
    devT = tf.keras.backend.std(y_true)
    
    pcc = tf.keras.backend.mean(fsp * fst) / (devP * devT)
    pcc = tf.where(tf.math.is_nan(pcc), 0.8, pcc)
    return pcc

# CNN Architecture
def create_model(input_size, lr=0.001):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(32, 4, 1, input_shape=input_size))
    model.add(tf.keras.layers.Activation("relu"))

    model.add(tf.keras.layers.Conv2D(64, 4, 1))
    model.add(tf.keras.layers.Activation("relu"))
    
    model.add(tf.keras.layers.Conv2D(128, 4, 1))
    model.add(tf.keras.layers.Activation("relu"))

    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(100, kernel_regularizer=tf.keras.regularizers.l2(0.01), ))
    model.add(tf.keras.layers.Activation("relu"))
    model.add(tf.keras.layers.Dropout(rate=args.rate))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Dense(50, kernel_regularizer=tf.keras.regularizers.l2(0.01), ))
    model.add(tf.keras.layers.Activation("relu"))
    model.add(tf.keras.layers.Dropout(rate=args.rate))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Dense(1, kernel_regularizer=tf.keras.regularizers.l2(0.01), ))
    model.add(tf.keras.layers.Activation("relu"))

    sgd = tf.keras.optimizers.SGD(lr=lr, momentum=0.9, decay=1e-6, clipvalue=args.clipvalue)
    model.compile(optimizer=sgd, loss=PCC_RMSE, metrics=["mse", PCC, RMSE, PCC_RMSE])
    
    return model

if __name__ == "__main__":
    print("Start training ...")

    d = """
    This architecture is constructed based on tensorflow2.3
    
    Train the CNN model through the extractes features of complexes in the training set.
    usage:
        python train.py -train_file train_features_pKa.csv -valid_file valid_features_pKa.csv
    """

    parser = argparse.ArgumentParser(description=d, formatter_class=RawDescriptionHelpFormatter)   

    parser.add_argument("-train_file", type=str, default="train_features_pKa.csv",
                        help="Input. This input file should include the features and pKa \n"
                             "of complexes in the training set.")
    parser.add_argument("-valid_file", type=str, default="valid_features_pKa.csv",
                        help="Input. This input file shuould include the features and pKa \n"
                             "of complexes in the validating set.")
    parser.add_argument("-shape", type=int, default=[84, 124, 1], nargs="+",
                        help="Input. Reshape the features.")
    parser.add_argument("-lr", type=float, default=0.001,
                        help="Input. The learning rate.")
    parser.add_argument("-batchsz", type=int, default=64,
                        help="Input. The number of samples processed per batch.")
    parser.add_argument("-rate", type=float, default=0.0,
                        help="Input. The dropout rate.")
    parser.add_argument("-alpha", type=float, default=0.7,
                        help="Input. The alpha value in loss function.")
    parser.add_argument("-clipvalue", type=float, default=0.01,
                        help="Input. The threshold for gradient clipping.")
    parser.add_argument("-n_features", type=int, default=10416,
                        help="Input. The number of features for each complex. \n"
                             "When shells N=62, n_feautes=21*8*62.")
    parser.add_argument("-epochs", type=int, default=300,
                        help="Input. The number of times all samples in the training set pass the CNN model.")
    parser.add_argument("-patience", type=int, default=30,
                        help="Input. Number of epochs with no improvement after which training will be stopped.")
    parser.add_argument("-out", type=str, default="bestmodel.h5",
                        help="output. The file path of saved best model. ")
    
    args = parser.parse_args()

    model = create_model(args.shape, args.lr)

    # load data
    train = pd.read_csv(args.train_file, index_col=0)
    print("Training set loaded ...")
    valid = pd.read_csv(args.valid_file, index_col=0)
    print("Validating set loaded ...")

    X_train = train.values[:, :args.n_features]
    X_valid = valid.values[:, :args.n_features]

    # Standardize the features
    scaler = preprocessing.StandardScaler()

    X_train_std = scaler.fit_transform(X_train).reshape([-1] + args.shape)
    X_valid_std = scaler.transform(X_valid).reshape([-1] + args.shape)

    joblib.dump(scaler, 'train_scaler.scaler')

    # labels
    y_train = train.pKa.values
    y_valid = valid.pKa.values

    # Callback
    stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.001, patience=args.patience,
                                            verbose=1, mode='auto', )
    logger = tf.keras.callbacks.CSVLogger("logfile", separator=',', append=False)
    bestmodel = tf.keras.callbacks.ModelCheckpoint(filepath=args.out, verbose=1, save_best_only=True)
        
    history = model.fit(X_train_std, y_train,
                        validation_data = (X_valid_std, y_valid),   
                        epochs = args.epochs,
                        batch_size = args.batchsz,
                        verbose=1,
                        callbacks=[stop, logger, bestmodel])
