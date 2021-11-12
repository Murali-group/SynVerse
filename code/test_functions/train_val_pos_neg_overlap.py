import pandas as pd
import pickle
import numpy as np
import random
import copy


main_dir = "/home/tasnina/Projects/SynVerse/outputs/cross_val/leave_comb/pairs_50_50000_th_0_cell_lines_20_percent_10_neg_1_neg_sampling_semi_random_val_frac_0.1_train_test_val2_finalv_1/"
run = 0
dir = main_dir+'run_'+str(run)+'/'

synergy_file = dir+"synergy.tsv"

non_synergy_file = dir+ 'non_synergy.tsv'

pos_train_test_val_file = dir +'pos_train_test_val.pkl'
neg_train_test_val_file = dir +'neg_train_test_val.pkl'

synergy_df = pd.read_csv(synergy_file, sep='\t', dtype={'Drug1_pubchem_cid': str,
                                                        'Drug2_pubchem_cid': str,
                                                        'Cell_line': str,
                                                        'Loewe': np.float64,
                                                        'Bliss': np.float64,
                                                        'ZIP': np.float64})

non_synergy_df = pd.read_csv(non_synergy_file, sep='\t', dtype={'Drug1_pubchem_cid': str,
                                                        'Drug2_pubchem_cid': str,
                                                        'Cell_line': str,
                                                        'Loewe_label': np.int32})
cell_lines = list(synergy_df['Cell_line'].unique())
with open(pos_train_test_val_file, 'rb') as handle:
     folds = pickle.load(handle)

with open(neg_train_test_val_file, 'rb') as handle:
    neg_folds = pickle.load(handle)



for fold in range(5):
    for split in ['train','test','val','val_es']:
        print('actual neg fact:', float(len(folds[fold][split]))/len(neg_folds[fold][split]))
        assert (float(len(folds[fold][split])) / len(neg_folds[fold][split]) <= 1 + 0.1) & \
               (float(len(folds[fold][split])) / len(neg_folds[fold][split]) >= 1 - 0.1)
# split_types = ['train', 'test','val','val_es']
# for fold in range(5):
#     drug_pairs={x:[] for x in split_types }
#     drug_pairs_pos = {x: [] for x in split_types}
#     drug_pairs_neg = {x: [] for x in split_types}
#     for split in split_types :
#         synergy_df_split = synergy_df[synergy_df.index.isin(folds[fold][split])]
#         non_synergy_df_split = non_synergy_df[non_synergy_df.index.isin(neg_folds[fold][split])]
#
#         # print(split, len(synergy_df_split), len(non_synergy_df_split))
#         drug_pairs_pos[split] = set(zip(synergy_df_split['Drug1_pubchem_cid'], synergy_df_split['Drug2_pubchem_cid']))
#         drug_pairs_neg[split] = set(zip(non_synergy_df_split['Drug1_pubchem_cid'], non_synergy_df_split['Drug2_pubchem_cid']))
#         drug_pairs[split] = drug_pairs_pos[split].union(drug_pairs_neg[split])
#
#     for i in range(4):
#         for j in range(i+1, 4):
#             print(split_types[i], split_types[j])
#
#             print(len(drug_pairs_pos[split_types[i]]), len(drug_pairs_pos[split_types[j]] ))
#             print( len(drug_pairs_pos[split_types[i]].intersection(drug_pairs_pos[split_types[j]])))
#
#
#             print(len(drug_pairs_neg[split_types[i]]), len(drug_pairs_neg[split_types[j]] ))
#             print( len(drug_pairs_neg[split_types[i]].intersection(drug_pairs_neg[split_types[j]])))
#
#             print(len(drug_pairs_pos[split_types[i]]), len(drug_pairs_neg[split_types[j]] ))
#             print( len(drug_pairs_pos[split_types[i]].intersection(drug_pairs_neg[split_types[j]])))
#
#             print(len(drug_pairs_neg[split_types[i]]), len(drug_pairs_pos[split_types[j]]))
#             print(len(drug_pairs_neg[split_types[i]].intersection(drug_pairs_pos[split_types[j]])))
#
#             print(len(drug_pairs[split_types[i]]), len(drug_pairs[split_types[j]]))
#             print(len(drug_pairs[split_types[i]].intersection(drug_pairs[split_types[j]])))
