#!/usr/bin/ python 3.X
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import itertools
from collections import OrderedDict
import re
from argparse import RawDescriptionHelpFormatter
import argparse

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
    
if __name__ == "__main__":

    print("Start Now ... ")

    d = """
        Generate the residue-atom contact features.

        The distance calculated in this script is the minimum distance between residues and atoms,\n
        that is, the distance between the atom (in the ligand) and the closest heavy atom in the specific residue.

        usage:
            python generate_features.py -inp input_complex.dat -out output_features.csv
        """

    parser = argparse.ArgumentParser(description=d, formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument("-inp", type=str, default="input_complex.dat",
                        help="Input. This file includes the path of the pdb file \n"
                             "for the protein-ligand complexes. \n"
                             "Each line in the dat file contains only one pdb file path.")
    parser.add_argument("-out", type=str, default="output_features.csv",
                        help="Output. The output file format is .csv \n"
                             "Each line in the output file contains only the features \n"
                             "of one protein-ligand complex.")
    parser.add_argument("-shells", type=int, default=62,
                        help="Input. The total number of shells. The optional range here \n"
                             "is 10 <= N <= 90.")

    args = parser.parse_args()
    with open(args.inp) as f:
        inputs = [x.strip() for x in f.readlines() if not x.startswith("#")]
    
    lig_defined_ele = ['H', 'C',  'O', 'N', 'P', 'S', 'Hal', 'DU']
    rec_defined_res = ['GLY', 'ALA', 'VAL', 'LEU', 'ILE', 'PRO', 'PHE', 'TYR', 'TRP', 'SER',
               'THR', 'CYS', 'MET', 'ASN', 'GLN', 'ASP', 'GLU', 'LYS', 'ARG', 'HIS', 'OTH']
    keys = ["_".join(x) for x in list(itertools.product(rec_defined_res, lig_defined_ele))]
    
    l = len(inputs)
    index = []
    values = []
    for i, inp in enumerate(inputs):
        pdb, rec_fpath, lig_fpath = inp.split()
        
        try:
            # load receptor
            rec = ParseProtein(rec_fpath)
            rec.parse_receptor()

            # load ligand 
            lig = ParseLigand(lig_fpath)
            lig.parse_ligand()

            # Generate features
            feat = GetFeatures(rec, lig, args.shells)
            result = feat.count_contacts()

            index.append(pdb)
            values.append(result.reshape(1, -1))
        
        except Exception as e:
            print("Error:", inp)
            print(e)
        
    values = np.concatenate(values, axis=0)
    columns = [f"{n}_{i}" for i, n in enumerate(keys * args.shells)]
    final_df = pd.DataFrame(values, index=index, columns=columns)
    final_df.to_pickle(args.out)