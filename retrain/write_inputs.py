import os

targets = os.listdir("../samples")

with open("inputs.dat", "w") as f:
    f.writelines(f"# id\trec_fpath\t\t\t\t\t\t\tlig_fpath\n")
    for t in targets:
        rec_fpath = f"../samples/{t}/{t}_protein.pdb"
        lig_fpath = f"../samples/{t}/{t}_ligand.pdb"
        f.writelines(f"{t}\t{rec_fpath}\t{lig_fpath}\n")