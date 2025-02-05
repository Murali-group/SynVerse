import pandas as pd
from utils import *
split_dir = "/home/grads/tasnina/Projects/SynVerse/inputs/splits/MARSY_data/D_ECFP_4_MACCS_MFP_d1hot_mol_graph_smiles_C_c1hot/k_0.05_S_mean_mean/leave_comb_0.2_0.25/run_0"
file_1 = f'{split_dir}/all_train.tsv'
file_2 = f'{split_dir}/test.tsv'

df_1 = pd.read_csv(file_1, sep='\t')[['drug_1_pid','drug_2_pid','cell_line_name']]
df_2 = pd.read_csv(file_2, sep='\t')[['drug_1_pid','drug_2_pid','cell_line_name']]


drug_pairs_1, drug_pids_1, cell_line_names_1 = print_synergy_stat(df_1)
drug_pairs_2, drug_pids_2, cell_line_names_2 = print_synergy_stat(df_2)


set_1 = set(zip(df_1['drug_1_pid'], df_1['drug_2_pid'], df_1['cell_line_name']))
set_2 = set(zip(df_2['drug_1_pid'], df_2['drug_2_pid'], df_2['cell_line_name']))



print('common tripletls:', len(set_1.intersection(set_2)))
print('union of triplets: ', len(set_1.union(set_2)))

print('common drugs:', len(drug_pids_1.intersection(drug_pids_2)))
print('common cell lines:', len(cell_line_names_1.intersection(cell_line_names_2)))

print('common drug pairs: ', len(set(drug_pairs_1).intersection(set(drug_pairs_2))))