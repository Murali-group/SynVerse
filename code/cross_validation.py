"""
The function for cross validation folds on training labels 
    - number of folds can be different
    - Type of cross validation: 1. Leave drug combination out in all cell line 2. Random Cross Validation 
                          3. Leave one-drug out 4.Leave one-cell line out
    - return split data in a list?




"""

import numpy as np
import time
import random
import pandas as pd
import os
from utils import negative_sampling
# def negative_sampling(cell_line, cell_line_spec_drug_drug_pairs_set_1, synergy_df):
#     # if (x,y) is in 'cell_line_spec_drug_drug_pairs_set', both (x,y) and (y,x) should not be in val_edges_false.
#     #that's why I used sorting here.
#     cell_line_spec_drug_drug_pairs_set = [(max(idx_1, idx_2), min(idx_1, idx_2)) for idx_1, idx_2 in \
#                                               cell_line_spec_drug_drug_pairs_set_1]
#
#     number_of_pairs_in_cell_line = len(cell_line_spec_drug_drug_pairs_set)
#
#     val_edges_false = set()
#     while True:
#         # idx_i = random.choices(synergy_df['Drug1_pubchem_cid'].unique(), k=int(number_of_pairs_in_cell_line))
#         # idx_j = random.choices(synergy_df['Drug2_pubchem_cid'].unique(), k=int(number_of_pairs_in_cell_line))
#         idx_i = list(synergy_df['Drug1_pubchem_cid'])
#
#         #change
#         idx_j = random.choices(synergy_df['Drug2_pubchem_cid'].unique(), k=int(number_of_pairs_in_cell_line))
#
#         new_val_edges = set(zip(idx_i, idx_j))
#
#         #sort
#         new_val_edges = set([(max(idx_1, idx_2), min(idx_1, idx_2)) for idx_1, idx_2 in new_val_edges])
#
#         new_false_val_edges = new_val_edges.difference(cell_line_spec_drug_drug_pairs_set)
#
#         val_edges_false = val_edges_false.union(new_false_val_edges)
#
#         if len(val_edges_false) >= int(number_of_pairs_in_cell_line):
#             # print('Val false edges done')
#             val_edges_false = np.array(list(val_edges_false)[0:int(number_of_pairs_in_cell_line)])
#             neg_df = pd.DataFrame({'Drug1_pubchem_cid': val_edges_false[:, 0], \
#                                    'Drug2_pubchem_cid': val_edges_false[:, 1], \
#                                    'Cell_line': [cell_line] * number_of_pairs_in_cell_line, \
#                                    'Loewe_label': [0] * number_of_pairs_in_cell_line})
#             break
#     assert len(neg_df) == len(cell_line_spec_drug_drug_pairs_set), 'problem negative sampling'
#     assert len(cell_line_spec_drug_drug_pairs_set_1) == len(cell_line_spec_drug_drug_pairs_set), 'problem data prep'
#
#     return neg_df

def is_neg_pos_dataset_non_overlapping(synergy_df, non_synergy_df):
    synergy_set = set(zip(synergy_df['Drug1_pubchem_cid'], synergy_df['Drug2_pubchem_cid'], synergy_df['Cell_line']))
    non_synergy_set = set(zip(non_synergy_df['Drug1_pubchem_cid'], non_synergy_df['Drug2_pubchem_cid'], non_synergy_df['Cell_line']))

    synergy_set = set([(max(drug1, drug2), min(drug1, drug2), cell_line) for drug1, drug2, cell_line in synergy_set ])
    non_synergy_set = set([(max(drug1, drug2), min(drug1, drug2), cell_line) for drug1, drug2, cell_line in non_synergy_set])

    common_between_syn_non_syn = synergy_set.intersection(non_synergy_set)
    if len(common_between_syn_non_syn) == 0:
        return True
    else:
        print(len(synergy_set), len(non_synergy_set),len(common_between_syn_non_syn), common_between_syn_non_syn)
        return False



def create_cross_val_folds(synergy_df, cross_val_type, number_of_folds, neg_fact):
    #this function will split the indexes in synergy_df into 'number_of_folds' according to\
    # the cross_val_type
    cell_lines = synergy_df['Cell_line'].unique()
    drugs = set(synergy_df['Drug1_pubchem_cid']).union(set(synergy_df['Drug2_pubchem_cid']))
    #find degree of durg in each cell_line

    if cross_val_type == 'leave_comb':
        t1 = time.time()
        pairs_per_fold = int(len(synergy_df) / number_of_folds + 1)
        drug_comb_count_dict = dict(synergy_df.groupby(['Drug1_pubchem_cid', 'Drug2_pubchem_cid'])\
                                ['Cell_line'].count())
        count = 0
        temp_synergy_df = synergy_df.copy()
        folds = {i: [] for i in range(number_of_folds)}
        for drug1, drug2 in drug_comb_count_dict:
            # print(drug1, drug2)
            df = temp_synergy_df[(temp_synergy_df['Drug1_pubchem_cid'] == drug1) &\
                            (temp_synergy_df['Drug2_pubchem_cid'] == drug2)]
            temp_synergy_df = temp_synergy_df[~temp_synergy_df.index.isin(list(df.index))]
            # print(count)
            while len(folds[count % number_of_folds]) >= pairs_per_fold: #this will skip a fold if it is already full
                count += 1 #go to next fold
            folds[count % number_of_folds] = folds[count % number_of_folds] + list(df.index)
            count += 1
        t2=time.time()

        for i in folds:
            print(str(i) + ': ', len(folds[i]))
        print(cross_val_type)
        print('time for creating folds for ', cross_val_type, 'time: ', str(t1-t2))
        return folds

    elif cross_val_type == 'leave_drug':

        drugs = list(set(synergy_df['Drug1_pubchem_cid'].astype(str)).union\
                                (set(synergy_df['Drug2_pubchem_cid'].astype(str))))
        folds = {}
        for drug in drugs:
            drug_df = synergy_df[(synergy_df['Drug1_pubchem_cid'] == drug) |\
                            (synergy_df['Drug2_pubchem_cid'] == drug)]

            # for some drugs the number of combinations is as low as 1.
            # so take only those drugs in test for which we have at least 100 pairs/combo available.
            if len(drug_df)>100:
                folds[drug] = list(drug_df.index)

        for i in folds:
            print(str(i) + ': ', len(folds[i]))
        print(cross_val_type)
        
    elif cross_val_type == 'leave_cell':


        folds = {cell_line: [] for cell_line in cell_lines}
        for cell_line in cell_lines:
            cell_line_df = synergy_df[synergy_df['Cell_line'] == cell_line]
            folds[cell_line] = list(cell_line_df.index)
        for i in folds:
            print(str(i) + ': ', len(folds[i]))
        print(cross_val_type)
        
    elif cross_val_type == 'random':
        # divide the drug-pairs from each cell line into n=number_of_folds folds. then combine the ith \
        # fold of each cell line to create overall ith fold.
        #this will give some notion of transfer learning i.e. how knowing synergy value of a comb in one cell\
        #can help to predict synergy is some other cell lines
        print('highest index', max(list(synergy_df.index)))
        # cell_lines = synergy_df['Cell_line'].unique()
        count = 0
        count_neg = 0
        folds = {i: [] for i in range(number_of_folds)}
        neg_folds = {i: [] for i in range(number_of_folds)}
        non_synergy_df = pd.DataFrame()
        for cell_line in cell_lines:

            init_df = synergy_df[synergy_df['Cell_line'] == cell_line]
            number_of_pairs_in_cell_line = len(init_df)

            df = synergy_df[synergy_df['Cell_line'] == cell_line]
            n_pairs_in_each_fold = int(number_of_pairs_in_cell_line/number_of_folds)
            # fold_choosing_order = np.arange(number_of_folds)
            # np.random.shuffle(fold_choosing_order)
            for ith_fold in range(number_of_folds):
                if len(df) > n_pairs_in_each_fold:
                    temp_df = df.sample(n_pairs_in_each_fold)
                else:
                    temp_df = df
                df = df[~df.index.isin(list(temp_df.index))]
                folds[ith_fold] = folds[ith_fold]+(list(temp_df.index))
            if len(df)>0:
                fold_no = count % number_of_folds
                folds[fold_no] = folds[fold_no] + (list(df.index))
                count += 1

            #do negative sampling
            # cell_line_spec_drug_drug_pairs_set = set(zip(init_df['Drug1_pubchem_cid'], init_df['Drug2_pubchem_cid']))
            neg_df = negative_sampling(cell_line, synergy_df, neg_fact)


            prev_len_of_no_syn_df = len(non_synergy_df)

            non_synergy_df = pd.concat([non_synergy_df, neg_df], axis=0)
            non_synergy_df.reset_index(drop=True, inplace=True)
            # print('non synergy: ', len(non_synergy_df) , max(non_synergy_df.index.values) + 1)
            assert len(non_synergy_df) == max(non_synergy_df.index.values) + 1, 'problem 100'
            start = prev_len_of_no_syn_df
            # end = len(non_synergy_df)
            for ith_fold in range(number_of_folds):
                end = start + n_pairs_in_each_fold
                neg_folds[ith_fold] = neg_folds[ith_fold] + list(non_synergy_df.index)[start: end]
                start = end
            if end < len(non_synergy_df):
                fold_no = count_neg % number_of_folds
                neg_folds[fold_no] = neg_folds[fold_no] + list(non_synergy_df.index)[end:len(non_synergy_df)]
                count_neg += 1

        print('pos folds')
        print('total: ', len(synergy_df))
        for i in folds:
            print(str(i) + ': ', len(folds[i]))

        print('neg_folds')
        print('total: ',len(non_synergy_df))
        for i in neg_folds:
            print(str(i) + ': ', len(neg_folds[i]))

        print(cross_val_type)

        assert is_neg_pos_dataset_non_overlapping(synergy_df, non_synergy_df) == True, 'problem in negative sampling in crossvalidation'
        return folds, neg_folds, non_synergy_df


def train_test_split(synergy_df,  test_frac):
    pairs_in_test_fold = int(len(synergy_df) * test_frac)
    test_index = random.choices(list(synergy_df.index), k = pairs_in_test_fold)
    test_df = synergy_df[synergy_df.index.isin(test_index)].reset_index()
    train_df = synergy_df[~synergy_df.index.isin(test_index)].reset_index()
    return train_df, test_df

# def train_test_split(synergy_df, cross_val_type,  test_frac, synergy_dir, force_run=False ):
#     train_file_name = synergy_dir +  cross_val_type + '/train.tsv'
#     test_file_name = synergy_dir +  cross_val_type + '/test.tsv'
#
#     if(~(os.path.exists(train_file_name) & os.path.exists(test_file_name)) | force_run==True):
#         test_df = pd.DataFrame()
#         if cross_val_type == 'leave_comb':
#             t1 = time.time()
#             pairs_in_test_fold = int(len(synergy_df)*test_frac)
#             drug_comb_count_dict = dict(synergy_df.groupby(['Drug1_pubchem_cid', 'Drug2_pubchem_cid']) \
#                                             ['Cell_line'].count())
#             # drug_comb_count_dict = dict(sorted(drug_comb_count_dict.items(), key=lambda x: x[1], reverse=True))
#             # print(drug_comb_count_dict.keys())
#             count = 0
#             temp_synergy_df = synergy_df.copy()
#             for drug1, drug2 in drug_comb_count_dict:
#                 # print(drug1, drug2)
#                 df = temp_synergy_df[(temp_synergy_df['Drug1_pubchem_cid'] == drug1) & \
#                                      (temp_synergy_df['Drug2_pubchem_cid'] == drug2)]
#                 temp_synergy_df = temp_synergy_df[~temp_synergy_df.index.isin(list(df.index))]
#                 # print(count)
#                 if len(test_df) < pairs_in_test_fold:
#                     test_df = pd.concat([test_df, df], axis=0)
#                 else:
#                     break
#
#         train_df = synergy_df[~synergy_df.index.isin(test_df.index)]
#         train_df.to_csv(train_file_name, sep='\t')
#         test_df.to_csv(test_file_name, sep='\t')
#     else:
#         train_df = pd.read_csv(train_file_name, sep='\t')
#         test_df = pd.read_csv(test_file_name, sep='\t')
#
#     return train_df, test_df