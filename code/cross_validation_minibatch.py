"""
The function for cross validation folds on training labels 
    - number of folds can be different
    - Type of cross validation: 1. Leave drug combination out in all cell line 2. Random Cross Validation 
                          3. Leave one-drug out 4.Leave one-cell line out
    - return split data in a list?


3. minibatch creation.

"""

import numpy as np
import time
import random
import pandas as pd
import os

def create_cross_val_folds(synergy_df, cross_val_type, number_of_folds):
    #this function will split the indexes in synergy_df into 'number_of_folds' according to\
    # the cross_val_type

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

        cell_lines = synergy_df['Cell_line'].unique()
        folds = {cell_line: [] for cell_line in cell_lines}
        for cell_line in cell_lines:
            cell_line_df = synergy_df[synergy_df['Cell_line'] == cell_line]
            folds[cell_line] = list(cell_line_df.index)
        for i in folds:
            print(str(i) + ': ', len(folds[i]))
        print(cross_val_type)
        
    elif cross_val_type == 'stratified':
        # divide the drug-pairs from each cell line into n=number_of_folds folds. then combine the ith \
        # fold of each cell line to create overall ith fold.
        #this will give some notion of transfer learning i.e. how knowing synergy value of a comb in one cell\
        #can help to predict synergy is some other cell lines
        print('highest index', max(list(synergy_df.index)))
        cell_lines = synergy_df['Cell_line'].unique()
        count = 0
        folds = {i: [] for i in range(number_of_folds)}
        for cell_line in cell_lines:
            df = synergy_df[synergy_df['Cell_line']==cell_line]
            n_pairs_in_each_fold = int(len(df)/number_of_folds)
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

        for i in folds:
            print(str(i) + ': ', len(folds[i]))
        print(cross_val_type)
        return folds


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