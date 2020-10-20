import sys
import os

code = sys.argv[1]

# Modify pdb file of the ligand.
# The residue name of the ligands is changed to "LIG".

with open(code + '_ligand.pdb') as f:
    lines = f.readlines()

with open(code + '_ligand_renamed.pdb', 'w') as f:
    for line in lines:
        if line[:4] in ['ATOM', 'HETA']:
            gro1 = line[:17]
            gro2 = 'LIG '
            gro3 = line[21:]
            f.writelines(gro1 + gro2 + gro3)

os.remove(code + '_ligand.pdb')

# Modify PDB file of the protein
# Only the rows containing 3D coordinates are kept.

inp_name = code + '_protein.pdb'
out_name = code + '_protein_atom.pdb'
if not os.path.exists(out_name):
    with open(inp_name, 'r') as f:
        with open(out_name, 'w') as a:
            for each in f:
                if 'ATOM' in each:
                    a.writelines(each)
                elif 'HETATM' in each:
                    a.writelines(each)
                else:
                    continue
