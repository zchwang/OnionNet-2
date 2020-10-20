#! /bin/bash

# Prepare the PDB file containing the 3D structure of protein-ligand complexes.

cd $1
dir_=$(ls $1)

for i in $dir_
do
	if [ `echo $i |wc -L` -eq 4 ];then
		
		cd $i
		echo $i
		
		if [ ! -f ${i}_ligand.pdb ];then
			obabel ${i}_ligand.mol2 -O ${i}_ligand.pdb
		fi
		
		if [ ! -f ${i}_ligand_renamed.pdb ] || [ ! -f ${i}_protein_atom.pdb ];then
			python ../../prepare/change_name.py ${i}
		fi
		
		cat ${i}_protein_atom.pdb ${i}_ligand_renamed.pdb > ${i}_cplx.pdb 
		
		rm ${i}_protein_atom.pdb
		rm ${i}_ligand_renamed.pdb

		cd ..
	fi
done
