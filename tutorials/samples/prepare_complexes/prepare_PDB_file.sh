#! /bin/bash

# Prepare the PDB file containing the 3D structure of protein-ligand complexes.

cd $1
dir_=$(ls)

for i in $dir_
do
	len=`expr length ${i}`
	if [ -d $i ] && [ ${len} -eq 4 ];then
		
		cd $i
		echo $i
		
		if [ ! -e ${i}_ligand.pdb ];then
			obabel ${i}_ligand.mol2 -O ${i}_ligand.pdb
			echo "${i}_ligand.pdb generated ..."
		fi
		
		if [ ! -e ${i}_ligand_renamed.pdb ] || [ ! -e ${i}_protein_atom.pdb ];then
			python ../../prepare/change_name.py ${i}
		fi
		
		cat ${i}_protein_atom.pdb ${i}_ligand_renamed.pdb > ${i}_cplx.pdb 
		
		rm ${i}_protein_atom.pdb
		rm ${i}_ligand_renamed.pdb

		cd ..
	fi
done
