import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.externals import joblib
from argparse import RawDescriptionHelpFormatter
import argparse

def PCC_RMSE(y_true, y_pred):
    global alpha
    fsp = y_pred - tf.keras.backend.mean(y_pred)
    fst = y_true - tf.keras.backend.mean(y_true)

    devP = tf.keras.backend.std(y_pred)
    devT = tf.keras.backend.std(y_true)

    rmse = tf.keras.backend.sqrt(tf.keras.backend.mean(tf.keras.backend.square(y_pred - y_true), axis=-1))

    pcc = 1.0 - tf.keras.backend.mean(fsp * fst) / (devP * devT)

    pcc = tf.where(tf.math.is_nan(pcc), 0.25, pcc)

    return alpha * pcc + (1 - alpha) * rmse

def RMSE(y_true, y_pred):
    return tf.keras.backend.sqrt(tf.keras.backend.mean(tf.keras.backend.square(y_pred - y_true), axis=-1))

def PCC(y_true, y_pred):
    fsp = y_pred - tf.keras.backend.mean(y_pred)
    fst = y_true - tf.keras.backend.mean(y_true)

    devP = tf.keras.backend.std(y_pred)
    devT = tf.keras.backend.std(y_true)
    return tf.keras.backend.mean(fsp * fst) / (devP * devT)

if __name__ == "__main__":

    d = """
        Predict the protein-ligand binding affinity.
        """
    parser = argparse.ArgumentParser(description=d, formatter_class=RawDescriptionHelpFormatter)

    parser.add_argument("-alpha", type=float, default=0.7,
                        help="Input. The alpha value in loss function.")
    parser.add_argument("-shape", type=int, default=[84, 124, 1], nargs="+",
                        help="Input. Reshape the features of the test set. When shells N = 62, shape = [-1, 84, 124, 1]."
                             "Note: The shape should be the same as the training and validation set.")
    parser.add_argument("-scaler", type=str, default="train_scaler.scaler",
                        help="Input. Load the .scaler file for preprocessing the features.")
    parser.add_argument("-model", type=str, default="bestmodel.h5",
                        help="Input. Load the model saved during training.")
    parser.add_argument("-inp", type=str, default="test_features.csv",
                        help="Input. The features of protein-ligand complexes in the test set.")
    parser.add_argument("-out", type=str, default="predicted_pKa.csv",
                        help="Output. The predicted pKa of the complex on the test set.")

    args = parser.parse_args()
    alpha = args.alpha

    test = pd.read_csv(args.inp, index_col=0)
    test_index = test.index
    
    X_test = test.values
    
    scaler = joblib.load(args.scaler)
    X_test_std = scaler.transform(X_test).reshape([-1] + args.shape)
    
    model = tf.keras.models.load_model(args.model,
            custom_objects={'RMSE': RMSE,
                'PCC': PCC,
                'PCC_RMSE': PCC_RMSE})

    pred_pKa = model.predict(X_test_std).ravel()

    pred_df = pd.DataFrame(pred_pKa, index=test_index, columns=['pKa'])
    pred_df.to_csv(args.out, float_format="%.3f")

