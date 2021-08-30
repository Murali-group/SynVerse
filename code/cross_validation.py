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
import copy
from utils import *
from utils import cell_line_based_negative_sampling_degree_based
from utils import cell_line_based_negative_sampling_semi_random
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

def is_train_test_val_non_overlapping(folds, df):
    for fold in folds:
        train_edges = df[df.index.isin(folds[fold]['train'])]
        test_edges = df[df.index.isin(folds[fold]['test'])]
        val_edges = df[df.index.isin(folds[fold]['val'])]

        train_edge_set = set(zip(train_edges['Drug1_pubchem_cid'], train_edges['Drug2_pubchem_cid'], train_edges['Cell_line']))
        train_edge_set = set([(max(drug1, drug2), min(drug1, drug2), cell_line) \
                              for drug1, drug2, cell_line in train_edge_set ])

        test_edge_set = set(zip(test_edges['Drug1_pubchem_cid'], test_edges['Drug2_pubchem_cid'], test_edges['Cell_line']))
        test_edge_set = set([(max(drug1, drug2), min(drug1, drug2), cell_line) for drug1, drug2, cell_line in
                          test_edge_set])
        val_edge_set = set(
            zip(val_edges['Drug1_pubchem_cid'], val_edges['Drug2_pubchem_cid'], val_edges['Cell_line']))
        val_edge_set = set([(max(drug1, drug2), min(drug1, drug2), cell_line) for drug1, drug2, cell_line in
                          val_edge_set])

        common_1 = set.intersection(train_edge_set,test_edge_set)
        common_2 = set.intersection(train_edge_set, val_edge_set)
        common_3 = set.intersection(val_edge_set, test_edge_set)

        if ((len(common_1)) > 0)| ((len(common_2)) > 0) | ((len(common_3)) > 0):
            print(len(common_1), len(common_2), len(common_3))
            print('common: ', common_1, common_2, common_3)
            return False

    return True


def create_cross_val_folds(synergy_df, cross_val_type, number_of_folds, neg_fact, neg_sampling_type):
    #this function will split the indexes in synergy_df into 'number_of_folds' according to\
    # the cross_val_type
    #return a dictionary of folds. key = fold_no, value= list of indices
    cell_lines = synergy_df['Cell_line'].unique()
    drugs = set(synergy_df['Drug1_pubchem_cid']).union(set(synergy_df['Drug2_pubchem_cid']))
    #find degree of durg in each cell_line

    # if cross_val_type == 'leave_comb':
    #     t1 = time.time()
    #     pairs_per_fold = int(len(synergy_df) / number_of_folds + 1)
    #     drug_comb_count_dict = dict(synergy_df.groupby(['Drug1_pubchem_cid', 'Drug2_pubchem_cid'])\
    #                             ['Cell_line'].count())
    #     count = 0
    #     temp_synergy_df = synergy_df.copy()
    #     folds = {i: [] for i in range(number_of_folds)}
    #     for drug1, drug2 in drug_comb_count_dict:
    #         # print(drug1, drug2)
    #         df = temp_synergy_df[(temp_synergy_df['Drug1_pubchem_cid'] == drug1) &\
    #                         (temp_synergy_df['Drug2_pubchem_cid'] == drug2)]
    #         temp_synergy_df = temp_synergy_df[~temp_synergy_df.index.isin(list(df.index))]
    #         # print(count)
    #         while len(folds[count % number_of_folds]) >= pairs_per_fold: #this will skip a fold if it is already full
    #             count += 1 #go to next fold
    #         folds[count % number_of_folds] = folds[count % number_of_folds] + list(df.index)
    #         count += 1
    #     t2=time.time()
    #
    #     for i in folds:
    #         print(str(i) + ': ', len(folds[i]))
    #     print(cross_val_type)
    #     # print('time for creating folds for ', cross_val_type, 'time: ', str(t1-t2))
    #     return folds
    #
    # elif cross_val_type == 'leave_drug':
    #
    #     drugs = list(set(synergy_df['Drug1_pubchem_cid'].astype(str)).union\
    #                             (set(synergy_df['Drug2_pubchem_cid'].astype(str))))
    #     folds = {}
    #     for drug in drugs:
    #         drug_df = synergy_df[(synergy_df['Drug1_pubchem_cid'] == drug) |\
    #                         (synergy_df['Drug2_pubchem_cid'] == drug)]
    #
    #         # for some drugs the number of combinations is as low as 1.
    #         # so take only those drugs in test for which we have at least 100 pairs/combo available.
    #         if len(drug_df)>100:
    #             folds[drug] = list(drug_df.index)
    #
    #     for i in folds:
    #         print(str(i) + ': ', len(folds[i]))
    #     print(cross_val_type)
    #
    # elif cross_val_type == 'leave_cell':
    #
    #
    #     folds = {cell_line: [] for cell_line in cell_lines}
    #     for cell_line in cell_lines:
    #         cell_line_df = synergy_df[synergy_df['Cell_line'] == cell_line]
    #         folds[cell_line] = list(cell_line_df.index)
    #     for i in folds:
    #         print(str(i) + ': ', len(folds[i]))
    #     print(cross_val_type)
        
    if cross_val_type == 'random':
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
            n_pairs_in_each_neg_fold = neg_fact*int(number_of_pairs_in_cell_line / number_of_folds)
            # fold_choosing_order = np.arange(number_of_folds)
            # np.random.shuffle(fold_choosing_order)

            for ith_fold in range(number_of_folds):
                if len(df) > n_pairs_in_each_fold:
                    temp_df = df.sample(n_pairs_in_each_fold)
                else:
                    temp_df = df
                df = df[~df.index.isin(list(temp_df.index))]
                folds[ith_fold] = folds[ith_fold]+(list(temp_df.index))

            if len(df) > 0:
                fold_no = count % number_of_folds
                folds[fold_no] = folds[fold_no] + (list(df.index))
                count += 1

            #do negative sampling
            # cell_line_spec_drug_drug_pairs_set = set(zip(init_df['Drug1_pubchem_cid'], init_df['Drug2_pubchem_cid']))

            # neg_df = negative_sampling_degree_dist(cell_line, synergy_df, neg_fact)
            if neg_sampling_type =='degree_based':
                neg_df = cell_line_based_negative_sampling_degree_based(cell_line, synergy_df, neg_fact)
            elif neg_sampling_type =='semi_random':
                neg_df = cell_line_based_negative_sampling_semi_random(cell_line, synergy_df, neg_fact)
            prev_len_of_no_syn_df = len(non_synergy_df)

            non_synergy_df = pd.concat([non_synergy_df, neg_df], axis=0)
            non_synergy_df.reset_index(drop=True, inplace=True)
            # print('non synergy: ', len(non_synergy_df) , max(non_synergy_df.index.values) + 1)
            assert len(non_synergy_df) == max(non_synergy_df.index.values) + 1, 'problem 100'
            start = prev_len_of_no_syn_df
            # end = len(non_synergy_df)
            for ith_fold in range(number_of_folds):
                end = start + n_pairs_in_each_neg_fold
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


def create_cross_val_folds_new(synergy_df, cross_val_type, number_of_folds, neg_fact, neg_sampling_type):

    cell_lines = synergy_df['Cell_line'].unique()
    if cross_val_type == 'random':
        # divide the drug-pairs from each cell line into n=number_of_folds folds. then combine the ith \
        # fold of each cell line to create overall ith fold.
        # this will give some notion of transfer learning i.e. how knowing synergy value of a comb in one cell\
        # can help to predict synergy is some other cell lines
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
            n_pairs_in_each_fold = int(number_of_pairs_in_cell_line / number_of_folds)
            n_pairs_in_each_neg_fold = neg_fact * int(number_of_pairs_in_cell_line / number_of_folds)
            # fold_choosing_order = np.arange(number_of_folds)
            # np.random.shuffle(fold_choosing_order)

            for ith_fold in range(number_of_folds):
                if len(df) > n_pairs_in_each_fold:
                    temp_df = df.sample(n_pairs_in_each_fold)
                else:
                    temp_df = df
                df = df[~df.index.isin(list(temp_df.index))]
                folds[ith_fold] = folds[ith_fold] + (list(temp_df.index))

            if len(df) > 0:
                fold_no = count % number_of_folds
                folds[fold_no] = folds[fold_no] + (list(df.index))
                count += 1

            # do negative sampling
            # cell_line_spec_drug_drug_pairs_set = set(zip(init_df['Drug1_pubchem_cid'], init_df['Drug2_pubchem_cid']))

            # neg_df = negative_sampling_degree_dist(cell_line, synergy_df, neg_fact)
            if neg_sampling_type == 'degree_based':
                neg_df = cell_line_based_negative_sampling_degree_based(cell_line, synergy_df, neg_fact)
            elif neg_sampling_type == 'semi_random':
                neg_df = cell_line_based_negative_sampling_semi_random(cell_line, synergy_df, neg_fact)
            prev_len_of_no_syn_df = len(non_synergy_df)
            weight = list(np.zeros(prev_len_of_no_syn_df))

            non_synergy_df = pd.concat([non_synergy_df, neg_df], axis=0)
            non_synergy_df.reset_index(drop=True, inplace=True)
            temp_non_synergy_df = copy.deepcopy(non_synergy_df)

            weight = weight + list(np.ones(len(neg_df)))
            # print('non synergy: ', len(non_synergy_df) , max(non_synergy_df.index.values) + 1)
            assert len(non_synergy_df) == max(non_synergy_df.index.values) + 1, 'problem 100'
            start = prev_len_of_no_syn_df
            # end = len(non_synergy_df)
            for ith_fold in range(number_of_folds):
                # end = start + n_pairs_in_each_neg_fold
                sampled_idx =  list(temp_non_synergy_df.sample(n=n_pairs_in_each_neg_fold,weights=weight).index)
                temp_non_synergy_df = temp_non_synergy_df[~temp_non_synergy_df.index.isin(sampled_idx)]
                weight = weight[0:-len(sampled_idx)]
                neg_folds[ith_fold] = neg_folds[ith_fold] + sampled_idx

                # start = end
            # if end < len(non_synergy_df):
            #     fold_no = count_neg % number_of_folds
            #     neg_folds[fold_no] = neg_folds[fold_no] + list(non_synergy_df.index)[end:len(non_synergy_df)]
            #     count_neg += 1

        print('pos folds')
        print('total: ', len(synergy_df))
        for i in folds:
            print(str(i) + ': ', len(folds[i]))

        print('neg_folds')
        print('total: ', len(non_synergy_df))
        for i in neg_folds:
            print(str(i) + ': ', len(neg_folds[i]))

        print(cross_val_type)

        assert is_neg_pos_dataset_non_overlapping(synergy_df,
                                                  non_synergy_df) == True, 'problem in negative sampling in crossvalidation'

        return folds, neg_folds, non_synergy_df


def create_test_and_train_cross_val_folds(synergy_df, cross_val_type,  number_of_folds, neg_fact, test_frac, neg_sampling_type):
    '''
    function: this function will split the indexes in synergy_df into test and train folds
    parameters:
        test_frac: fraction of test edges in total edges in synergy_df
        number_of_folds: number of folds to create in training dataset

    return: dictionaries and lists and datafarme. in dictionaries, key = fold_no, value= list of indices
            folds = dict of train indices of positive edges from synergy_df
            neg_folds = dict of train indices of negative edges from non_synergy_df,
            test_fold = list of  test indices of positive edges from synergy_df
            test_neg_fold = list of test indices of negative edges from non_synergy_df
            non_synergy_df = negative sampled cell_line_specific_non_synergistic drug pairs.
                            this df has similar columns as synergy_df
    '''


    cell_lines = synergy_df['Cell_line'].unique()
    # drugs = set(synergy_df['Drug1_pubchem_cid']).union(set(synergy_df['Drug2_pubchem_cid']))
    # find degree of durg in each cell_line

    if cross_val_type == 'leave_comb':
        t1 = time.time()
        pairs_per_fold = int(len(synergy_df) / number_of_folds + 1)
        drug_comb_count_dict = dict(synergy_df.groupby(['Drug1_pubchem_cid', 'Drug2_pubchem_cid']) \
                                        ['Cell_line'].count())
        count = 0
        temp_synergy_df = copy.deepcopy(synergy_df)
        folds = {i: [] for i in range(number_of_folds)}
        for drug1, drug2 in drug_comb_count_dict:
            # print(drug1, drug2)
            df = temp_synergy_df[(temp_synergy_df['Drug1_pubchem_cid'] == drug1) & \
                                 (temp_synergy_df['Drug2_pubchem_cid'] == drug2)]
            temp_synergy_df = temp_synergy_df[~temp_synergy_df.index.isin(list(df.index))]
            # print(count)
            while len(folds[count % number_of_folds]) >= pairs_per_fold:  # this will skip a fold if it is already full
                count += 1  # go to next fold
            folds[count % number_of_folds] = folds[count % number_of_folds] + list(df.index)
            count += 1
        t2 = time.time()

        for i in folds:
            print(str(i) + ': ', len(folds[i]))
        print(cross_val_type)
        # print('time for creating folds for ', cross_val_type, 'time: ', str(t1-t2))
        return folds

    elif cross_val_type == 'leave_drug':

        drugs = list(set(synergy_df['Drug1_pubchem_cid'].astype(str)).union \
                         (set(synergy_df['Drug2_pubchem_cid'].astype(str))))
        folds = {}
        for drug in drugs:
            drug_df = synergy_df[(synergy_df['Drug1_pubchem_cid'] == drug) | \
                                 (synergy_df['Drug2_pubchem_cid'] == drug)]

            # for some drugs the number of combinations is as low as 1.
            # so take only those drugs in test for which we have at least 100 pairs/combo available.
            if len(drug_df) > 100:
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
        # this will give some notion of transfer learning i.e. how knowing synergy value of a comb in one cell\
        # can help to predict synergy is some other cell lines
        print('highest index', max(list(synergy_df.index)))
        # cell_lines = synergy_df['Cell_line'].unique()
        count = 0
        count_neg = 0
        test_fold = []
        test_neg_fold = []
        folds = {i: [] for i in range(number_of_folds)}
        neg_folds = {i: [] for i in range(number_of_folds)}
        non_synergy_df = pd.DataFrame()

        for cell_line in cell_lines:

            init_df = synergy_df[synergy_df['Cell_line'] == cell_line]
            df = synergy_df[synergy_df['Cell_line'] == cell_line]  #can I just do deepcopy?

            number_of_pairs_in_cell_line = len(init_df)
            n_pairs_in_test_fold = int(number_of_pairs_in_cell_line * test_frac)
            n_pairs_in_each_training_fold = int((number_of_pairs_in_cell_line - n_pairs_in_test_fold) / number_of_folds)

            n_pairs_in_test_neg_fold = neg_fact*int(number_of_pairs_in_cell_line * test_frac)
            n_pairs_in_each_training_neg_fold = neg_fact*int((number_of_pairs_in_cell_line - n_pairs_in_test_fold) / number_of_folds)

            test_fold = test_fold + list(df.sample(n_pairs_in_test_fold).index)
            df = df[~df.index.isin(test_fold)]

            for ith_fold in range(number_of_folds):
                if len(df) > n_pairs_in_each_training_fold:
                    temp_df = df.sample(n_pairs_in_each_training_fold)
                else:
                    temp_df = df
                df = df[~df.index.isin(list(temp_df.index))]
                folds[ith_fold] = folds[ith_fold] + (list(temp_df.index))
            if len(df) > 0:
                fold_no = count % number_of_folds
                folds[fold_no] = folds[fold_no] + (list(df.index))
                count += 1

            # do negative sampling
            if neg_sampling_type =='degree_based':
                neg_df = cell_line_based_negative_sampling_degree_based(cell_line, synergy_df, neg_fact)
            elif neg_sampling_type =='semi_random':
                neg_df = cell_line_based_negative_sampling_semi_random(cell_line, synergy_df, neg_fact)

            prev_len_of_no_syn_df = len(non_synergy_df)

            non_synergy_df = pd.concat([non_synergy_df, neg_df], axis=0)
            non_synergy_df.reset_index(drop=True, inplace=True)
            # print('non synergy: ', len(non_synergy_df) , max(non_synergy_df.index.values) + 1)
            assert len(non_synergy_df) == max(non_synergy_df.index.values) + 1, 'problem 100'
            start = prev_len_of_no_syn_df
            # end = len(non_synergy_df)

            end = start + n_pairs_in_test_neg_fold
            test_neg_fold = test_neg_fold + list(non_synergy_df.index)[start: end]

            start = end

            for ith_fold in range(number_of_folds):
                end = start + n_pairs_in_each_training_neg_fold
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
        print('total: ', len(non_synergy_df))
        for i in neg_folds:
            print(str(i) + ': ', len(neg_folds[i]))

        print(cross_val_type)

        assert is_neg_pos_dataset_non_overlapping(synergy_df,
                                                  non_synergy_df) == True, 'problem in negative sampling in crossvalidation'
        return folds, neg_folds, test_fold, test_neg_fold, non_synergy_df


def leave_comb_cross_val(synergy_df, number_of_folds, val_frac):
    drug_comb_count_dict = dict(synergy_df.groupby(['Drug1_pubchem_cid', 'Drug2_pubchem_cid']) \
                                    ['Cell_line'].count())

    init_drug_combs = set(drug_comb_count_dict.keys())
    n_drug_comb_in_test = int(len(init_drug_combs) / number_of_folds)
    n_drug_comb_in_val = int(len(init_drug_combs) * val_frac)

    folds = {i: {'test': [], 'train': [], 'val': [], 'val_es': []} for i in range(number_of_folds)}
    # neg_folds = {i: {'test': [], 'train': [], 'val': []} for i in range(number_of_folds)}

    drug_combs = set(drug_comb_count_dict.keys())
    for fold in range(number_of_folds):
        folds[fold]['test'] = random.sample(drug_combs, n_drug_comb_in_test)
        drug_combs = drug_combs.difference(folds[fold]['test'])

    drug_combs = set(drug_comb_count_dict.keys())
    for fold in range(number_of_folds):
        drug_combs = drug_combs.difference(folds[fold]['test'])
        folds[fold]['val'] = random.sample(drug_combs, n_drug_comb_in_val)

        drug_combs = drug_combs.difference(folds[fold]['val'])
        folds[fold]['val_es'] = random.sample(drug_combs, n_drug_comb_in_val)

        drug_combs = drug_combs.difference(folds[fold]['val_es'])
        folds[fold]['train'] = copy.deepcopy(drug_combs)

        drug_combs = set(drug_comb_count_dict.keys())

        assert len(set(folds[fold]['train']).intersection(set(folds[fold]['val']))) == 0, print('overalpping drugs btn train and val')
        assert len(set(folds[fold]['train']).intersection(set(folds[fold]['test']))) == 0, print('overalpping drugs btn train and test')
        assert len(set(folds[fold]['val']).intersection(set(folds[fold]['test']))) == 0, print('overalpping drugs btn val and test')

        assert len(set(folds[fold]['train']).intersection(set(folds[fold]['val_es']))) == 0, print('overalpping drugs btn train and val_es')
        assert len(set(folds[fold]['test']).intersection(set(folds[fold]['val_es']))) == 0, print('overalpping drugs btn test and val_es')
        assert len(set(folds[fold]['val']).intersection(set(folds[fold]['val_es']))) == 0, print('overalpping drugs btn val and val_es')

    for fold in range(number_of_folds):
        for split_type in ['train', 'test', 'val', 'val_es']:
            folds[fold][split_type] = list(synergy_df[
                synergy_df[['Drug1_pubchem_cid', 'Drug2_pubchem_cid']].apply(tuple, axis=1).isin(
                    folds[fold][split_type])].index)

            print('fold no: %d  split type: %s', fold, split_type)
            print('n pairs: %d', len(folds[fold][split_type]))

    return folds

def leave_drug_cross_val(synergy_df, number_of_folds, val_frac):
    drugs = set(synergy_df['Drug1_pubchem_cid'].astype(str)).union \
        (set(synergy_df['Drug2_pubchem_cid'].astype(str)))
    init_drugs = copy.deepcopy(drugs)
    n_drugs_in_test = int(len(drugs) / number_of_folds)
    n_drugs_in_val = int(len(drugs) * val_frac)

    folds = {i: {'test': [], 'train': [], 'val': [], 'val_es':[]} for i in range(number_of_folds)}

    # neg_folds = {i: {'test': [], 'train': [], 'val': []} for i in range(number_of_folds)}
    for fold in range(number_of_folds):
        folds[fold]['test'] = random.sample(drugs, n_drugs_in_test)
        drugs = drugs.difference(folds[fold]['test'])

    drugs = copy.deepcopy(init_drugs)
    for fold in range(number_of_folds):
        drugs = drugs.difference(folds[fold]['test'])
        folds[fold]['val'] = random.sample(drugs, n_drugs_in_val)

        drugs = drugs.difference(folds[fold]['val'])
        folds[fold]['val_es'] = random.sample(drugs, n_drugs_in_val)

        drugs = drugs.difference(folds[fold]['val_es'])
        folds[fold]['train'] = copy.deepcopy(drugs)
        drugs = copy.deepcopy(init_drugs)

        assert len(set(folds[fold]['train']).intersection(set(folds[fold]['val']))) == 0, print('overalpping drugs btn train and val')
        assert len(set(folds[fold]['train']).intersection(set(folds[fold]['test']))) == 0, print('overalpping drugs btn train and test')
        assert len(set(folds[fold]['val']).intersection(set(folds[fold]['test']))) == 0, print('overalpping drugs btn test and val')

        assert len(set(folds[fold]['train']).intersection(set(folds[fold]['val_es']))) == 0, print('overalpping drugs btn train and val_es')
        assert len(set(folds[fold]['test']).intersection(set(folds[fold]['val_es']))) == 0, print('overalpping drugs btn test and val_es')
        assert len(set(folds[fold]['val']).intersection(set(folds[fold]['val_es']))) == 0, print('overalpping drugs btn val and val_es')

    for fold in range(number_of_folds):
        for split_type in ['train', 'test', 'val', 'val_es']:
            folds[fold][split_type] = list(synergy_df[
                                               (synergy_df['Drug1_pubchem_cid'].isin(folds[fold][split_type])) & \
                                               (synergy_df['Drug2_pubchem_cid'].isin(folds[fold][split_type]))].index)
            print('fold no: %d  split type: %s', fold, split_type)
            print('n pairs: %d', len(folds[fold][split_type]))
    return folds

def create_test_val_train_cross_val_folds(synergy_df, init_non_synergy_df, cross_val_type,  number_of_folds, neg_fact, val_frac, neg_sampling_type):
    '''
    function: this function will split the indexes in synergy_df into number of folds.
    parameters:
        val_frac: fraction of val edges in total edges in synergy_df
        number_of_folds: number of folds to create in training dataset

    return: dictionaries and lists and datafarme. in dictionaries, key = fold_no, value= list of indices
            folds = dict of train indices of positive edges from synergy_df
            neg_folds = dict of train indices of negative edges from non_synergy_df,
            non_synergy_df = negative sampled cell_line_specific_non_synergistic drug pairs.
                            this df has similar columns as synergy_df
    '''

    cell_lines = synergy_df['Cell_line'].unique()

    if cross_val_type == 'leave_drug':

        folds = leave_drug_cross_val(synergy_df, number_of_folds, val_frac)
        #do negative sampling
        if neg_sampling_type == 'degree_based':
            neg_folds, non_synergy_df = drug_based_negative_sampling_degree_based(folds, synergy_df, neg_fact, number_of_folds)
        elif neg_sampling_type == 'semi_random':
            neg_folds, non_synergy_df = drug_based_negative_sampling_semi_random(folds, synergy_df, neg_fact, number_of_folds)
        elif neg_sampling_type == 'no':
            non_synergy_df = init_non_synergy_df
            neg_folds = leave_drug_cross_val(init_non_synergy_df, number_of_folds, val_frac)

    elif cross_val_type == 'leave_comb':
        folds = leave_comb_cross_val(synergy_df, number_of_folds, val_frac)

        non_synergy_df=pd.DataFrame()
        for cell_line in cell_lines:
            if neg_sampling_type == 'degree_based':
                neg_df = cell_line_based_negative_sampling_degree_based(cell_line, synergy_df, neg_fact)
                non_synergy_df = pd.concat([non_synergy_df, neg_df], axis=0)
                print('neg cell line: ', cell_line)
            elif neg_sampling_type == 'semi_random':
                neg_df = cell_line_based_negative_sampling_semi_random(cell_line, synergy_df, neg_fact)
                non_synergy_df = pd.concat([non_synergy_df, neg_df], axis=0)
                print('neg cell line: ', cell_line)

            elif neg_sampling_type == 'no':
                non_synergy_df = init_non_synergy_df
                break

        non_synergy_df.reset_index(drop=True, inplace= True)
        neg_folds = leave_comb_cross_val(non_synergy_df, number_of_folds, val_frac)

    elif cross_val_type == 'random':

        count = 0
        count_neg = 0
        folds = {i: {'test': [], 'train': [], 'val': [], 'val_es':[]} for i in range(number_of_folds)}
        neg_folds = {i: {'test': [], 'train': [], 'val': [], 'val_es':[]} for i in range(number_of_folds)}

        for cell_line in cell_lines:
            init_df = synergy_df[synergy_df['Cell_line'] == cell_line]
            number_of_pairs_in_cell_line = len(init_df)
            df = synergy_df[synergy_df['Cell_line'] == cell_line]
            n_pairs_in_each_fold = int(number_of_pairs_in_cell_line / number_of_folds)

            for ith_fold in range(number_of_folds):
                if len(df) > n_pairs_in_each_fold:
                    temp_df = df.sample(n_pairs_in_each_fold)
                else:
                    temp_df = df
                df = df[~df.index.isin(list(temp_df.index))]
                folds[ith_fold]['test'] = folds[ith_fold]['test'] + list(temp_df.index)

            if len(df) > 0:
                fold_no = count % number_of_folds
                folds[fold_no]['test'] = folds[fold_no]['test'] + list(df.index)
                count += 1

            #for each fold i (consider it as a test fold), sample val_frac edges from all other folds except i as validation edges.

            for i in range(number_of_folds):
                df = copy.deepcopy(init_df)

                #remove test indices from df
                df = df[~df.index.isin(folds[i]['test'])]

                #now sample 'val' from remaining df
                val_df = df.sample(frac=val_frac)
                #now remove val indices from df
                df = df[~df.index.isin(val_df.index)]

                # now sample 'val_es' i.e. validation dataset for early stopping from remaining df
                val_es_df = df.sample(frac=val_frac)
                # now remove val_es indices from df
                df = df[~df.index.isin(val_es_df.index)]

                train_df = copy.deepcopy(df)

                folds[i]['val'] = folds[i]['val'] + list(val_df.index)
                folds[i]['val_es'] = folds[i]['val_es'] + list(val_es_df.index)
                folds[i]['train'] = folds[i]['train'] + list(train_df.index)

            print('pos cell line: ', cell_line)

        #create non_synergy_df
        non_synergy_df = pd.DataFrame()

        for cell_line in cell_lines:
            if neg_sampling_type == 'degree_based':
                neg_df = cell_line_based_negative_sampling_degree_based(cell_line, synergy_df, neg_fact)
                non_synergy_df = pd.concat([non_synergy_df, neg_df], axis=0)
                print('neg cell line: ', cell_line)

            elif neg_sampling_type == 'semi_random':
                neg_df = cell_line_based_negative_sampling_semi_random(cell_line, synergy_df, neg_fact)
                non_synergy_df = pd.concat([non_synergy_df, neg_df], axis=0)
                print('neg cell line: ', cell_line)
            elif neg_sampling_type == 'no':
                non_synergy_df = init_non_synergy_df
                break

        non_synergy_df.reset_index(drop=True, inplace=True)

        for cell_line in cell_lines:
            n_pairs_in_each_neg_fold = neg_fact * int(number_of_pairs_in_cell_line / number_of_folds)
            init_df = non_synergy_df[non_synergy_df['Cell_line'] == cell_line]
            number_of_pairs_in_cell_line = len(init_df)
            df = non_synergy_df[non_synergy_df['Cell_line'] == cell_line]

            for ith_fold in range(number_of_folds):
                if len(df) > n_pairs_in_each_neg_fold:
                    temp_df = df.sample(n_pairs_in_each_neg_fold)
                else:
                    temp_df = df
                df = df[~df.index.isin(list(temp_df.index))]
                neg_folds[ith_fold]['test'] = neg_folds[ith_fold]['test'] + list(temp_df.index)

            if len(df) > 0:
                fold_no = count % number_of_folds
                neg_folds[fold_no]['test'] = neg_folds[fold_no]['test'] + list(df.index)
                count += 1

            # for each fold i (consider it as a test fold), sample val_frac edges from all other folds except i as validation edges.
            for i in range(number_of_folds):
                df = copy.deepcopy(init_df)
                # remove test indices from df
                df = df[~df.index.isin(neg_folds[i]['test'])]

                # now sample 'val' from remaining df
                val_df = df.sample(frac=val_frac)
                # now remove val indices from df
                df = df[~df.index.isin(val_df.index)]

                # now sample 'val_es' i.e. validation dataset for early stopping from remaining df
                val_es_df = df.sample(frac=val_frac)
                # now remove val_es indices from df
                df = df[~df.index.isin(val_es_df.index)]

                train_df = copy.deepcopy(df)

                neg_folds[i]['val'] = neg_folds[i]['val'] + list(val_df.index)
                neg_folds[i]['val_es'] = neg_folds[i]['val_es'] + list(val_es_df.index)
                neg_folds[i]['train'] = neg_folds[i]['train'] + list(train_df.index)

        print('pos folds')
        print('total: ', len(synergy_df))
        for i in folds:
            print(str(i) + ': ', len(folds[i]['test']),  len(folds[i]['train']),  len(folds[i]['val']))

        print('neg_folds')
        print('total: ', len(non_synergy_df))
        for i in neg_folds:
            print(str(i) + ': ', len(neg_folds[i]['test']), len(neg_folds[i]['train']), len(neg_folds[i]['val']))

        print(cross_val_type)

        assert is_neg_pos_dataset_non_overlapping(synergy_df,
                                                  non_synergy_df) == True, 'problem in negative sampling in crossvalidation'

    assert is_train_test_val_non_overlapping(folds, synergy_df) == True, 'problem in train-test-val split of positive edges'
    assert is_train_test_val_non_overlapping(neg_folds,
                                             non_synergy_df) == True, 'problem in train-test-val split of negative edges'
    return folds, neg_folds, non_synergy_df


#
# synergy_file = "/home/tasnina/Projects/SynVerse/inputs/synergy/synergy_labels.tsv"
# synergy_df = pd.read_csv(synergy_file, sep='\t', dtype={'Drug1_pubchem_cid': str,
#                                                         'Drug2_pubchem_cid': str,
#                                                         'Cell_line': str,
#                                                         'Loewe': np.float64,
#                                                         'Bliss': np.float64,
#                                                         'ZIP': np.float64})
# print(synergy_df.nunique())
# synergy_df = synergy_df.sample(1000)
# create_test_val_train_cross_val_folds(synergy_df, 'random',  5, 1, 0.1, 'semi_random')
# print('hello')