import numpy as np
import pandas as pd
from argparse import RawDescriptionHelpFormatter
import argparse


if __name__ == "__main__":
    print("Start Now ...")
    d = """
        The features and the real pKa are concated in a common file as the input of the training process.

        Usage:
            python concat_features_pKa.py -inp_features output_features.csv
        -inp_true all_complexes_pKa.csv -out output_features_pKa.csv
        """
    parser = argparse.ArgumentParser(description=d, formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument("-inp_features", type=str, default="output_features.csv",
                        help="Input. Specify the features.")
    parser.add_argument("-inp_true", type=str, default="all_complexes_pKa.csv",
                        help="Input. Specify all complexes pKa.")
    parser.add_argument("-out", type=str, default="output_features_pKa.csv",
                        help="Output. Specify the target file name.")
    args = parser.parse_args()
    
    features = pd.read_csv(args.inp_features, index_col=0)
    index = features.index.tolist()
    values = features.values

    dict_ = {}
    for k, v in zip(index, values):
        dict_[k] = v.tolist()

    
    true = pd.read_csv(args.inp_true, index_col=0)
    true_index = true.index.tolist()
    true_values = true.values

    true_dict = {}
    for k, v in zip(true_index, true_values):
        true_dict[k] = v.tolist()

    new_values = []
    for i in index:
        new_values.append(dict_[i] + true_dict[i])
    print(len(new_values))
    print(len(new_values[0]))
    columns = features.columns.tolist() + ['pKa']

df = pd.DataFrame(np.array(new_values), index = index, columns = columns)
df.to_csv(args.out)


