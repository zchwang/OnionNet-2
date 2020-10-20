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
        
        self.lig_indices = np.array([])
        self.all_pairs = []
        self.atom_pairs = []
        
        self.masses = np.array([])
        self.atoms_indices = []
        self.residues_indices = []
        self.residues = []
        self.lig_ele = np.array([])
        
        self.xyz = np.array([])
        self.MCs = []
    
        self.residue_atom_distances = np.array([])
        self.counts_ = np.array([])
        
    def parsePDB(self):
        top = self.pdb.topology

        # Number of all groups in the complex except the ligand
        self.residues_indices = [i for i in range(top.n_residues - 1)]

        # Get the serial number of the atom in each residue or group
        for i in self.residues_indices:
            each_atoms = [j.index for j in top.residue(i).atoms]
            self.atoms_indices.append(each_atoms)

        # Get the masses of atoms
        self.masses = np.array([atom.element.mass for atom in top.atoms])

        # Get the 3D coordinates for all atoms
        self.xyz = self.pdb.xyz[0]

        # residues names
        self.residues = [str(residue)[:3] for residue in top.residues][:-1]
        
        # Get the serial number of each atom in the ligand
        self.lig_indices = top.select('resname LIG')

        # Get the types of atoms in the ligand
        table, bond = top.to_dataframe()
        self.lig_ele = table['element'][self.lig_indices]
        
        return self

    # Calculate the centroid of the residue
    def calculate_MC(self):
        self.parsePDB()
           
        for i in self.residues_indices:
            atom_indices = self.atoms_indices[i]
            mass_xyz = [self.masses[j] * self.xyz[j] for j in atom_indices]
            residue_mass = np.sum(np.array([self.masses[g] for g in atom_indices]))
            MC = np.sum(mass_xyz, axis=0)/residue_mass
            self.MCs.append(MC)
        return self

    # Calculate the distance between residues in the protein and atoms in the ligand
    def compute_distances(self):
        self.calculate_MC()
        self.all_pairs = list(itertools.product(self.residues_indices, self.lig_indices))
        distances = []
        for residue, atom in self.all_pairs:
            
            distance = np.sqrt(np.sum(np.square(np.array(self.MCs[residue]) - self.xyz[atom])))
            
            distances.append(distance)
        self.residue_atom_distances = np.array(distances)
        return self

    def cutoff_count(self, cutoff):
        self.counts_ = (self.residue_atom_distances <= cutoff) * 1
        return self

# Define all residue types
all_residues = ['GLY', 'ALA', 'VAL', 'LEU', 'ILE', 'PRO', 'PHE', 'TYR', 'TRP', 'SER',
               'THR', 'CYS', 'MET', 'ASN', 'GLN', 'ASP', 'GLU', 'LYS', 'ARG', 'HIS', 'HETATM']
def get_residue(residue):
    if residue in all_residues:
        return residue
    else:
        return 'HETATM'

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

def generate_features(fn, ncutoffs):
    cplx = AtomType(fn)   
    cplx.compute_distances()
    
    residue_names = []
    lig_atom_names = []

    # Types of the residue and the atom 
    for num, i in enumerate(cplx.all_pairs):
        residue_name = cplx.residues[int(i[0])]
        lig_atom_name = cplx.lig_ele[int(i[1])]
        residue_names.append(get_residue(residue_name))
        lig_atom_names.append(get_elementtype(lig_atom_name))

    # residue-atom pairs    
    residues_lig_atoms_combines = ["_".join(x) for x in zip(residue_names, lig_atom_names)]

    # calculate the number of contacts in different shells
    counts = []
    onion_counts = []
    for i, cutoff in enumerate(ncutoffs):
        cplx.cutoff_count(cutoff)
    
        if i == 0:
            onion_counts.append(cplx.counts_)
        else:
            onion_counts.append(cplx.counts_ - counts[-1])
        counts.append(cplx.counts_)

    results = []
    
    for n in range(len(ncutoffs)):
        d = OrderedDict()
        d = d.fromkeys(keys, 0)
        
        for k, v in zip(residues_lig_atoms_combines, onion_counts[n]):
            d[k] += v
        results += d.values()
    return results

if __name__ == "__main__":

    print("Start Now ... ")

    d = """
        Generate the residue-atom contact features.
        
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
                             "is 16 <= N <= 90.")

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
