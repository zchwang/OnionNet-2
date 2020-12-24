#!/usr/bin/ python 3.X
import numpy as np
import pandas as pd
from argparse import RawDescriptionHelpFormatter
import argparse
import mdtraj as md
import itertools
from collections import OrderedDict

class AtomType():
    def __init__(self, fn):
        self.pdb = md.load(fn)
        self.rec_indices = np.array([])
        self.lig_indices = np.array([])
        self.all_pairs = []
        self.atom_pairs = []
        
        self.masses = np.array([])
        self.atoms_indices = []
        self.residues_indices = []
        self.residues = []
        self.all_ele = np.array([])
        self.lig_ele = np.array([])
        
        self.xyz = np.array([])
        self.distances = np.array([])
        
        self.counts_ = np.array([])
        
    def parsePDB(self):
        top = self.pdb.topology
        residues = [str(residue)[:3] for residue in top.residues]
        residues_cp = residues.copy()

        # number of all groups in the complex except the ligand
        LIG_number = residues.index('LIG')
        self.residues_indices = [i for i in range(top.n_residues) if i != LIG_number]

        # Get the types of atoms in the protein and ligand
        self.rec_indices = top.select('protein')
        self.lig_indices = top.select('resname LIG')
        table, bond = top.to_dataframe()
        self.all_ele = table['element']
        self.lig_ele = table['element'][self.lig_indices]
        
        H_num = []
        for num, i in enumerate(self.all_ele):
            if i == 'H':
                H_num.append(num)

        # Get the serial number of the atom in each residue or group
        removes = []
        for i in self.residues_indices:
            atoms = top.residue(i).atoms
            each_atoms = [j.index for j in atoms]
            heavy_atoms = [x for x in each_atoms if not x in H_num]
            
            if len(heavy_atoms) == 0:
                removes.append(i)
            else:
                self.atoms_indices.append(heavy_atoms)
        
        if len(removes) != 0:
            for i in removes:
                self.residues_indices.remove(i)
        self.residues = [residues_cp[x] for x in self.residues_indices]

        # Get the 3D coordinates for all atoms
        self.xyz = self.pdb.xyz[0]
        
        return self

    # Calculate the minimum distance between reisdues in the protein and atoms in the ligand
    def compute_distances(self):
        self.parsePDB()
        distances = []

        for r_atom in self.atoms_indices:
            if len(r_atom) == 0:
                continue

            for l_atom in self.lig_indices:
                ds = []
                for i in r_atom: 
                    d = np.sqrt(np.sum(np.square(self.xyz[i] - self.xyz[l_atom])))
                    ds.append(d)
                distances.append(min(ds))

        self.distances = np.array(distances)
    
        return self
    
    def cutoff_count(self, distances, cutoff):
        self.counts_ = (self.distances <= cutoff) * 1
        return self

# Define all residue types
all_residues = ['GLY', 'ALA', 'VAL', 'LEU', 'ILE', 'PRO', 'PHE', 'TYR', 'TRP', 'SER',
               'THR', 'CYS', 'MET', 'ASN', 'GLN', 'ASP', 'GLU', 'LYS', 'ARG', 'HIS', 'OTH']
def get_residue(residue):
    if residue in all_residues:
        return residue
    else:
        return 'OTH'

# Define all element types
all_elements = ['H', 'C',  'O', 'N', 'P', 'S', 'Hal', 'DU']
Hal = ['F', 'Cl', 'Br', 'I']
def get_elementtype(e):

    if e in all_elements:
        return e
    elif e in Hal:
        return 'Hal'
    else:
        return 'DU'

# all residue-atom combination pairs
keys = ["_".join(x) for x in list(itertools.product(all_residues, all_elements))]

def generate_features(fn, cutoffs):
    cplx = AtomType(fn)
    cplx.compute_distances()

    # Types of the residue and the atom 
    new_residue = list(map(get_residue, cplx.residues))
    new_lig = list(map(get_elementtype, cplx.lig_ele))

    # residue-atom pairs
    residues_lig_atoms_combines = ["_".join(x) for x in list(itertools.product(new_residue, new_lig))]

    # calculate the number of contacts in different shells
    counts = []
    onion_counts = []

    for i, cutoff in enumerate(cutoffs):
        cplx.cutoff_count(cplx.distances, cutoff)
        counts_ = cplx.counts_
        if i == 0:
            onion_counts.append(counts_)
        else:
            onion_counts.append(counts_ - counts[-1])
        counts.append(counts_)
    results = []

    for n in range(len(cutoffs)):
        d = OrderedDict()
        d = d.fromkeys(keys, 0)

        for e_e, c in zip(residues_lig_atoms_combines, onion_counts[n]):
            d[e_e] += c
        results += d.values()
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
        lines = f.readlines()
    inputs = [x[:-1] for x in lines]

    results = []
    index = []

    outermost = 0.05 * (args.shells + 1)
    ncutoffs = np.linspace(0.1, outermost, args.shells)

    l = len(inputs)
    for i, fn in enumerate(inputs):
        result = generate_features(fn, ncutoffs)
        results.append(result)
        index.append(fn)
        print(fn, i, l)

    columns = []
    for i, n in enumerate(keys * len(ncutoffs)):
        columns.append(n + '_' + str(i))

    df = pd.DataFrame(np.array(results), index=index, columns=columns)
    df.to_csv(args.out, float_format='%.1f')
