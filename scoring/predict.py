import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn import preprocessing
import joblib
from scipy.spatial.distance import cdist
import itertools
from collections import OrderedDict
import re
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

class ParseProtein():
    def __init__(self, rec_fpath):
        with open(rec_fpath) as f:
            self.lines = [x.strip() for x in f.readlines() if x[:4] in ["ATOM", "HETA"]]
        self.defined_res = ['GLY', 'ALA', 'VAL', 'LEU', 'ILE', 'PRO', 'PHE', 'TYR', 'TRP', 'SER',
               'THR', 'CYS', 'MET', 'ASN', 'GLN', 'ASP', 'GLU', 'LYS', 'ARG', 'HIS', 'OTH']
        #self.rec_ele_list = []
        self.all_res_xyz_list = []
        self.res_list = []
          
    def get_res(self, res):
        if res in self.defined_res:
            return res
        else:
            return "OTH"
        
    def extract_letter(self, ele):
        pattern = re.compile(r'([A-Za-z]+)\d+[+-]')
        match = pattern.match(ele)
        if match:
            letter_part = match.group(1)
            return letter_part
        else:
            return ele
            
    def parse_receptor(self): 
        sym_pool = []
        num = -1      
        _temp_res_xyz = []
        for line in self.lines:
            ele = line.split()[-1]
            ele = self.extract_letter(ele)
            if ele == "H":
                continue
            num += 1
            
            res = line[17:20].strip()
            res = self.get_res(res)
            sym = line[17:27].strip()
            x = float(line[30:38].strip())
            y = float(line[38:46].strip())
            z = float(line[46:54].strip())
            if num == 0:
                self.res_list.append(res)
                sym_pool.append(sym)
                _temp_res_xyz.append([x, y, z])
            else:
                #if res == self.res_list[-1]:
                if sym == sym_pool[-1]:
                    _temp_res_xyz.append([x, y, z])
                else:
                    self.all_res_xyz_list.append(np.array(_temp_res_xyz) * 0.1)
                    _temp_res_xyz = [[x, y, z]]
                    sym_pool.append(sym)
                    self.res_list.append(res)
                    
        self.all_res_xyz_list.append(np.array(_temp_res_xyz) * 0.1)
        return self

class ParseLigand():
    def __init__(self, lig_fpath):
        with open(lig_fpath) as f:
            self.lines = [x.strip() for x in f.readlines() if x[:4] in ["ATOM", "HETA"]]
        self.defined_eles = ['H', 'C',  'O', 'N', 'P', 'S', 'Hal', 'DU']
        self.hal_ele = ["F", "Cl", "Br", "I"]
        self.lig_ele_list = []
        self.lig_xyz_array = np.array([])
    
    def get_ele(self, ele):
        if ele in self.defined_eles:
            return ele
        elif ele in self.hal_ele:
            return "Hal"
        else:
            return "DU"
        
    def extract_letter(self, ele):
        pattern = re.compile(r'([A-Za-z]+)\d+[+-]')
        match = pattern.match(ele)
        if match:
            letter_part = match.group(1)
            return letter_part
        else:
            return ele
            
    def parse_ligand(self):
        lig_xyz = []
        for line in self.lines:
            x = float(line[30:38].strip())
            y = float(line[38:46].strip())
            z = float(line[46:54].strip())
            ele = line.split()[-1]
            ele = self.extract_letter(ele)
            ele = self.get_ele(ele)
            self.lig_ele_list.append(ele)
            lig_xyz.append([x, y, z])
        
        self.lig_xyz_array = np.array(lig_xyz) * 0.1
        
        return self
    
class GetFeatures():
    def __init__(self, rec, lig, shell):
        self.rec = rec
        self.lig = lig
        self.N_shell = shell
        self.res_atom_pairs = []
        self.res_atom_dist = []
        
    def cal_distance(self):
        res_atom_dist = []
        for res, res_xyz in zip(self.rec.res_list, self.rec.all_res_xyz_list):
            for ele, atom_xyz in zip(self.lig.lig_ele_list, self.lig.lig_xyz_array):
                pair = f"{res}_{ele}"
                dist_mtx = cdist(atom_xyz.reshape(1, -1), res_xyz, metric='euclidean')
                self.res_atom_pairs.append(pair)
                res_atom_dist.append(dist_mtx.min())
        self.res_atom_dist = np.array(res_atom_dist)
        return self
    
    def count_contacts(self):
        self.cal_distance()
        
        outermost = 0.05 * (self.N_shell + 1)
        ncutoffs = np.linspace(0.1, outermost, self.N_shell)
        
        temp_counts = []
        onion_counts = []
        for i, cutoff in enumerate(ncutoffs):
            _contact_bool = (self.res_atom_dist <= cutoff) * 1
            if i == 0:
                onion_counts.append(_contact_bool)
            else:
                onion_counts.append(_contact_bool - temp_counts[-1])
            temp_counts.append(_contact_bool)
        temp_counts = []
        
        results = []
        for n in range(len(ncutoffs)):
            d = OrderedDict()
            d = d.fromkeys(keys, 0)
            for e_e, c in zip(self.res_atom_pairs, onion_counts[n]):
                d[e_e] += c
            results.append(np.array(list(d.values())).ravel())
        results = np.concatenate(results, axis=0)
        return results
    
def generate_features(rec_fpath, lig_fpath):
    # load receptor
    rec = ParseProtein(rec_fpath)
    rec.parse_receptor()

    # load ligand 
    lig = ParseLigand(lig_fpath)
    lig.parse_ligand()

    # Generate features
    feat = GetFeatures(rec, lig, args.shells)
    result = feat.count_contacts()

    return result

if __name__ == "__main__":

    d = """
        Predict the protein-ligand binding affinity.
        """
    parser = argparse.ArgumentParser(description=d, formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument("-rec_fpath", type=str, default="protein.pdb",
                        help="Input. PDB format, the file of receptor.")
    parser.add_argument("-lig_fpath", type=str, default="ligand.pdb",
                        help="Input. PDB/SDF/MOL2 format, the file of receptor.")
    parser.add_argument("-alpha", type=float, default=0.7,
                        help="Input. The alpha value in loss function.")
    #parser.add_argument("-shape", type=int, default=[84, 124, 1], nargs="+",
    #                    help="Input. Reshape the features of the test set. When shells N = 62, shape = [-1, 84, 124, 1]."
    #                         "Note: The shape should be the same as the training and validation set.")
    parser.add_argument("-shape", type=str, default="84,124,1",
                        help="Input. Reshape the features.")
    parser.add_argument("-scaler", type=str, default="train_scaler.scaler",
                        help="Input. Load the .scaler file for preprocessing the features.")
    parser.add_argument("-model", type=str, default="bestmodel.h5",
                        help="Input. Load the model saved during training.")
    parser.add_argument("-shells", type=int, default=62,
                        help="Input. The total number of shells.")
    parser.add_argument("-out_fpath", type=str, default="predicted_pKd.csv",
                        help="Output. The predicted pKa of the complex on the test set.")
    args = parser.parse_args()
    rec_fpath = args.rec_fpath
    lig_fpath = args.lig_fpath
    shape = shape = [int(x) for x in args.shape.split(",")]

    lig_defined_ele = ['H', 'C',  'O', 'N', 'P', 'S', 'Hal', 'DU']
    rec_defined_res = ['GLY', 'ALA', 'VAL', 'LEU', 'ILE', 'PRO', 'PHE', 'TYR', 'TRP', 'SER',
               'THR', 'CYS', 'MET', 'ASN', 'GLN', 'ASP', 'GLU', 'LYS', 'ARG', 'HIS', 'OTH']
    keys = ["_".join(x) for x in list(itertools.product(rec_defined_res, lig_defined_ele))]
    X_feat = generate_features(rec_fpath, lig_fpath)
    scaler = joblib.load(args.scaler)
    X_test_std = scaler.transform(X_feat.reshape(1, -1)).reshape([-1] + shape)

    
    model = tf.keras.models.load_model(args.model,
            custom_objects={'RMSE': RMSE,
                'PCC': PCC,
                'PCC_RMSE': PCC_RMSE})

    pred_pKa = model.predict(X_test_std).ravel()

    pred_df = pd.DataFrame(pred_pKa, columns=['pred_pKd'])
    pred_df.to_csv(args.out_fpath, float_format="%.3f")