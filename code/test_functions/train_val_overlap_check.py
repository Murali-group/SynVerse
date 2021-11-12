import pandas as pd
import pickle
import numpy as np
import random
import copy



synergy_file = '/home/tasnina/Projects/SynVerse/outputs/cross_val/random/\
pairs_50_50000_th_0_cell_lines_20_percent_10_neg_1_semi_random_neg_sampling_test_train_genex_1_testingphase1/run_0/synergy.tsv'
pos_train_test_val_file = '/home/tasnina/Projects/SynVerse/outputs/cross_val/random/\
pairs_50_50000_th_0_cell_lines_20_percent_10_neg_1_semi_random_neg_sampling_test_train_genex_1_testingphase1/run_0/pos_train_test_val.pkl'
neg_train_test_val_file = '/home/tasnina/Projects/SynVerse/outputs/cross_val/random/\
pairs_50_50000_th_0_cell_lines_20_percent_10_neg_1_semi_random_neg_sampling_test_train_genex_1_testingphase1/run_0/neg_train_test_val.pkl'

synergy_df = pd.read_csv(synergy_file, index_col=0, sep='\t', dtype={'Drug1_pubchem_cid': str,
                                                        'Drug2_pubchem_cid': str,
                                                        'Cell_line': str,
                                                        'Loewe': np.float64,
                                                        'Bliss': np.float64,
                                                        'ZIP': np.float64})
cell_lines = list(synergy_df['Cell_line'].unique())
# with open(pos_train_test_val_file, 'rb') as handle:
#      pos_dict= pickle.load(handle)
# # print(pos_dict)
#
# print(pos_dict.keys())
# print('initial edges: ', len(synergy_df))


drugs = set(synergy_df['Drug1_pubchem_cid'].astype(str)).union \
                 (set(synergy_df['Drug2_pubchem_cid'].astype(str)))
init_drugs = copy.deepcopy(drugs)
n_folds = 5
val_frac = 0.1
drugs_in_each_fold = int(len(drugs)/n_folds)
drugs_in_val = int(len(drugs)*val_frac)

pos_dict = {i: {'test': [], 'train': [], 'val': []} for i in range(n_folds)}
for fold in range(5):
    pos_dict[fold]['test'] = random.sample(drugs, drugs_in_each_fold)
    drugs = drugs.difference(pos_dict[fold]['test'])

    pos_dict[fold]['val'] = random.sample(drugs, drugs_in_val)
    drugs = drugs.difference(pos_dict[fold]['val'])

    pos_dict[fold]['train'] = copy.deepcopy(drugs)

    del drugs
    drugs = copy.deepcopy(init_drugs)

for fold in range(5):
    for split_type in ['train', 'test', 'val']:
        pos_dict[fold][split_type] = synergy_df[(synergy_df['Drug1_pubchem_cid'].isin(pos_dict[fold][split_type])) &\
                             (synergy_df['Drug2_pubchem_cid'].isin(pos_dict[fold][split_type]))].index
        print('fold no: %d  split type: %s', fold, split_type)
        print('n pairs: %d',  len(pos_dict[fold][split_type] ))



# for drug in drugs:
#     drug_df = synergy_df[(synergy_df['Drug1_pubchem_cid'] == drug) | \
#                          (synergy_df['Drug2_pubchem_cid'] == drug)]
#
#     print('drug_name', drug, type(drug))
#     print('number of edges for the drug: ', len(drug_df))




for fold in range(1):

    print('**************FOLD**********************', str(fold))
    train_df = synergy_df[synergy_df.index.isin(pos_dict[fold]['train'])]
    train_df.to_csv('train_'+str(fold)+'.csv')

    test_df = synergy_df[synergy_df.index.isin(pos_dict[fold]['test'])]
    test_df.to_csv('test_' + str(fold) + '.csv')

    val_df = synergy_df[synergy_df.index.isin(pos_dict[fold]['val'])]
    val_df.to_csv('val_' + str(fold) + '.csv')
    for cell_line in cell_lines:
        df_val = val_df[val_df['Cell_line']==cell_line]
        d1_d2_val = set(zip(list(df_val['Drug1_pubchem_cid']),list(df_val['Drug2_pubchem_cid'])))
        d1_d2_val = set([(max(d1,d2), min(d1, d2)) for (d1, d2) in d1_d2_val])

        # print(d1_d2_val)

        df_train = train_df[train_df['Cell_line']==cell_line]

        d1_d2_train = set(zip(list(df_train['Drug1_pubchem_cid']), list(df_train['Drug2_pubchem_cid'])))
        d1_d2_train = set([(max(d1, d2), min(d1, d2)) for (d1, d2) in d1_d2_train])

        print('cell_line: ', cell_line)

        print('number of d1_d2 pair in train dataset       : ', len(train_df[train_df['Cell_line']==cell_line]))

        print('number of d1_d2 pair in test dataset       : ', len(test_df[test_df['Cell_line'] == cell_line]))

        print('number of d1_d2 pair in validation dataset       : ', len(d1_d2_val))
        print('number of common d1_d2 pair between train and val: ', len(d1_d2_val.intersection(d1_d2_train)))
        # print('common: ', d1_d2_val.intersection(d1_d2_train))


