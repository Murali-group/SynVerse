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

def is_train_test_val_val_es_non_overlapping(folds, df):
    for fold in folds:
        train_edges = df[df.index.isin(folds[fold]['train'])]
        test_edges = df[df.index.isin(folds[fold]['test'])]
        val_edges = df[df.index.isin(folds[fold]['val'])]
        val_es_edges = df[df.index.isin(folds[fold]['val_es'])]

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

        val_es_edge_set = set(
            zip(val_es_edges['Drug1_pubchem_cid'], val_es_edges['Drug2_pubchem_cid'], val_es_edges['Cell_line']))
        val_es_edge_set = set([(max(drug1, drug2), min(drug1, drug2), cell_line) for drug1, drug2, cell_line in
                            val_es_edge_set])

        common_1 = set.intersection(train_edge_set,test_edge_set)
        common_2 = set.intersection(train_edge_set, val_edge_set)
        common_3 = set.intersection(val_edge_set, test_edge_set)
        common_4 = set.intersection(train_edge_set, val_es_edge_set)
        common_5 = set.intersection(test_edge_set, val_es_edge_set)
        common_6 = set.intersection(val_edge_set, val_es_edge_set)

        if ((len(common_1)) > 0)| ((len(common_2)) > 0) | ((len(common_3)) > 0)\
            |((len(common_4)) > 0)| ((len(common_5)) > 0) | ((len(common_6)) > 0):

            print(len(common_1), len(common_2), len(common_3),len(common_4), len(common_5), len(common_6))
            # print('common: ', common_1, common_2, common_3)
            return False

    return True


def create_drug_pair_folds(synergy_df, number_of_folds, val_frac):

    init_drug_combs = set(zip(synergy_df['Drug1_pubchem_cid'],synergy_df['Drug2_pubchem_cid']))
    n_drug_comb_in_test = int(len(init_drug_combs) / number_of_folds)
    n_drug_comb_in_val = int(len(init_drug_combs) * val_frac)

    drug_pair_folds = {i: {'test': [], 'train': [], 'val': [], 'val_es': []} for i in range(number_of_folds)}

    drug_combs = copy.deepcopy(init_drug_combs)
    for fold in range(number_of_folds):
        drug_pair_folds[fold]['test'] = random.sample(drug_combs, n_drug_comb_in_test )
        drug_combs = drug_combs.difference(drug_pair_folds[fold]['test'])

    drug_combs = copy.deepcopy(init_drug_combs)
    for fold in range(number_of_folds):
        drug_combs = drug_combs.difference(drug_pair_folds[fold]['test'])
        drug_pair_folds[fold]['val'] = random.sample(drug_combs, n_drug_comb_in_val)

        drug_combs = drug_combs.difference(drug_pair_folds[fold]['val'])
        drug_pair_folds[fold]['val_es'] = random.sample(drug_combs, n_drug_comb_in_val)

        drug_combs = drug_combs.difference(drug_pair_folds[fold]['val_es'])
        drug_pair_folds[fold]['train'] = list(copy.deepcopy(drug_combs))

        drug_combs = copy.deepcopy(init_drug_combs)

        assert len(set(drug_pair_folds[fold]['train']).intersection(set(drug_pair_folds[fold]['val']))) == 0, print(
            'overalpping drugs btn train and val')
        assert len(set(drug_pair_folds[fold]['train']).intersection(set(drug_pair_folds[fold]['test']))) == 0, print(
            'overalpping drugs btn train and test')
        assert len(set(drug_pair_folds[fold]['val']).intersection(set(drug_pair_folds[fold]['test']))) == 0, print(
            'overalpping drugs btn val and test')
        assert len(set(drug_pair_folds[fold]['train']).intersection(set(drug_pair_folds[fold]['val_es']))) == 0, print(
            'overalpping drugs btn train and val_es')
        assert len(set(drug_pair_folds[fold]['test']).intersection(set(drug_pair_folds[fold]['val_es']))) == 0, print(
            'overalpping drugs btn test and val_es')
        assert len(set(drug_pair_folds[fold]['val']).intersection(set(drug_pair_folds[fold]['val_es']))) == 0, print(
            'overalpping drugs btn val and val_es')

    return drug_pair_folds

def leave_comb_cross_val(synergy_df, number_of_folds, val_frac, prev_folds=None):
    drug_pair_folds = create_drug_pair_folds(synergy_df, number_of_folds, val_frac)
    folds = {i: {'test': [], 'train': [], 'val': [], 'val_es': []} for i in range(number_of_folds)}
    for fold in range(number_of_folds):
        for split_type in ['train', 'test', 'val', 'val_es']:

            folds[fold][split_type] = list(synergy_df[
                synergy_df[['Drug1_pubchem_cid', 'Drug2_pubchem_cid']].apply(tuple, axis=1).isin(
                    drug_pair_folds[fold][split_type])].index)

            print('fold no: %d  split type: %s', fold, split_type)
            print('n pairs: %d', len(folds[fold][split_type]))

    if prev_folds!=None:
        #so this is for no_synegry_df
        for fold in range(number_of_folds):
            for split_type in ['train', 'test', 'val', 'val_es']:
                folds[fold][split_type] = random.sample(folds[fold][split_type], len(prev_folds[fold][split_type]))


    return folds, drug_pair_folds


def leave_cell_line_cross_val(synergy_df, number_of_folds, val_frac, cell_lines_per_fold=None):
    init_cell_lines = synergy_df['Cell_line'].unique()

    n_cell_lines_in_test = int(len(init_cell_lines) / number_of_folds)
    n_cell_lines_in_val = int(len(init_cell_lines) * val_frac)

    if cell_lines_per_fold == None:
        cell_lines_per_fold={i: {'test': [], 'train': [], 'val': [], 'val_es': []} for i in range(number_of_folds)}
        cell_lines = set(init_cell_lines)
        for fold in range(number_of_folds):
            cell_lines_per_fold[fold]['test'] = random.sample(cell_lines, n_cell_lines_in_test)
            cell_lines = cell_lines.difference(cell_lines_per_fold[fold]['test'])

        cell_lines = set(init_cell_lines)
        for fold in range(number_of_folds):
            cell_lines = cell_lines.difference(cell_lines_per_fold[fold]['test'])
            cell_lines_per_fold[fold]['val'] = random.sample(cell_lines, n_cell_lines_in_val)

            cell_lines = cell_lines.difference(cell_lines_per_fold[fold]['val'])
            cell_lines_per_fold[fold]['val_es'] = random.sample(cell_lines, n_cell_lines_in_val)

            cell_lines = cell_lines.difference(cell_lines_per_fold[fold]['val_es'])
            cell_lines_per_fold[fold]['train'] = copy.deepcopy(cell_lines)

            cell_lines = set(init_cell_lines)

            assert len(set(cell_lines_per_fold[fold]['train']).intersection(set(cell_lines_per_fold[fold]['val']))) == 0, print('overalpping drugs btn train and val')
            assert len(set(cell_lines_per_fold[fold]['train']).intersection(set(cell_lines_per_fold[fold]['test']))) == 0, print('overalpping drugs btn train and test')
            assert len(set(cell_lines_per_fold[fold]['val']).intersection(set(cell_lines_per_fold[fold]['test']))) == 0, print('overalpping drugs btn val and test')

            assert len(set(cell_lines_per_fold[fold]['train']).intersection(set(cell_lines_per_fold[fold]['val_es']))) == 0, print('overalpping drugs btn train and val_es')
            assert len(set(cell_lines_per_fold[fold]['test']).intersection(set(cell_lines_per_fold[fold]['val_es']))) == 0, print('overalpping drugs btn test and val_es')
            assert len(set(cell_lines_per_fold[fold]['val']).intersection(set(cell_lines_per_fold[fold]['val_es']))) == 0, print('overalpping drugs btn val and val_es')

    folds = {i: {'test': [], 'train': [], 'val': [], 'val_es': []} for i in range(number_of_folds)}
    for fold in range(number_of_folds):
        for split_type in ['train', 'test', 'val', 'val_es']:
            folds[fold][split_type] = list(synergy_df[
                synergy_df['Cell_line'].isin(
                    cell_lines_per_fold[fold][split_type])].index)

            print('fold no: %d  split type: %s', fold, split_type)
            print('n pairs: %d', len(folds[fold][split_type]))

    #test how many drug-pairs (nt considering which cell lines they are from) \
    # are overlapping between train-test folds
    for fold in range(number_of_folds):
        df_train = synergy_df[synergy_df.index.isin(folds[fold]['train'])][['Drug1_pubchem_cid','Drug2_pubchem_cid']]
        df_test = synergy_df[synergy_df.index.isin(folds[fold]['test'])][['Drug1_pubchem_cid', 'Drug2_pubchem_cid']]

        train_pairs = set(zip(df_train['Drug1_pubchem_cid'], df_train['Drug2_pubchem_cid']))
        test_pairs = set(zip(df_test['Drug1_pubchem_cid'], df_test['Drug2_pubchem_cid']))

        train_pairs = set([(max(x,y),min(x,y)) for x,y in train_pairs])
        test_pairs = set([(max(x, y), min(x, y)) for x, y in test_pairs])

        common_train_test  = len(train_pairs.intersection(test_pairs))
        print('fold_no: ', fold, len(train_pairs), len(test_pairs), common_train_test)


    return folds, cell_lines_per_fold



def leave_comb_cell_line_cross_val(synergy_df_init, num_folds, val_frac, prev_folds=None):
    n_pairs_in_test = int(len(synergy_df_init)/num_folds)
    n_pairs_in_val = int(len(synergy_df_init)*val_frac)
    folds = {i: {'test': [], 'train': [], 'val': [], 'val_es':[]} for i in range(num_folds)}

    for i in range(num_folds):
        synergy_df = copy.deepcopy(synergy_df_init)
        while True:
            # print('remaining cell lines: ', synergy_df['Cell_line'].unique())
            drug_cell_line_triplets = list(zip(synergy_df['Drug1_pubchem_cid'], synergy_df['Drug2_pubchem_cid'],
                        synergy_df['Cell_line']))
            drug_cell_line_triplet = random.sample(drug_cell_line_triplets, 1)
            drug_pairs = [(x, y) for x, y, z in drug_cell_line_triplet]
            cell_lines = [z for x, y, z in drug_cell_line_triplet]

            # find all the index of synergy_df for which drug_pairs\
            # or cell line matches with the sampled drug_cell_line_triplet
            df = synergy_df[synergy_df[['Drug1_pubchem_cid', 'Drug2_pubchem_cid']].
                             apply(tuple, axis=1).isin(drug_pairs)]
            df1 = synergy_df[synergy_df['Cell_line'].isin(cell_lines)]
            df = pd.concat([df,df1], axis=0)
            print('n triplets: ', len(df))
            # print('selected cell line: ', df['Cell_line'].unique())

            # selected_triplets = set(zip(df['Drug1_pubchem_cid'], df['Drug2_pubchem_cid'], df['Cell_line']))
            buffer = 50
            if len(folds[i]['test']) < n_pairs_in_test-buffer:
                folds[i]['test'] = folds[i]['test'] + list((df.index))
                print('test: ', 'n triplets: ', len(df))
            elif len(folds[i]['val']) < n_pairs_in_val-buffer:
                folds[i]['val'] = folds[i]['val'] + list((df.index))
                print('val: ', 'n triplets: ', len(df))
            elif len(folds[i]['val_es']) < n_pairs_in_val-buffer:
                folds[i]['val_es'] = folds[i]['val_es'] + list((df.index))
                print('val_es: ', 'n triplets: ', len(df))
            else:
                folds[i]['train'] = list(synergy_df.index)
                print('train: ', 'n triplets: ', len(synergy_df))
                break
            synergy_df = synergy_df[~synergy_df.index.isin(list(df.index))]

        for split in ['train','test','val','val_es']:
            l = synergy_df_init[synergy_df_init.index.isin(folds[i][split])]['Cell_line'].unique()
            print('fold: ', i, 'split: ', split, l)

    if prev_folds!=None:
        #so this is for no_synegry_df
        for fold in range(num_folds):
            for split_type in ['train', 'test', 'val', 'val_es']:
                folds[fold][split_type] = random.sample(folds[fold][split_type], len(prev_folds[fold][split_type]))

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

def create_rare_cell_line_split(synergy_df, init_non_synergy_df, n_rare_cell_lines, number_of_folds, neg_fact,val_frac, neg_sampling_type):
    cell_lines = synergy_df['Cell_line'].unique()
    #do negative sampling
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


    #keeping the split data in similar format as previous cross validations
    folds = {i: {'test': [], 'train': [], 'val': [], 'val_es': []} for i in range(number_of_folds)}
    neg_folds = {i: {'test': [], 'train': [], 'val': [], 'val_es': []} for i in range(number_of_folds)}
    #count which cell line has how many drug_pairs involved
    cell_line_wise_n_pairs_dict = dict(synergy_df.groupby(by=['Cell_line'])['Drug1_pubchem_cid'].count())
    cell_line_wise_n_pairs_dict = dict(sorted(cell_line_wise_n_pairs_dict.items(), key=lambda x: x[1], reverse=True))
    sorted_cell_lines = list(cell_line_wise_n_pairs_dict.keys())

    train_val_es_cell_lines = sorted_cell_lines[0:-n_rare_cell_lines]
    test_cell_lines = sorted_cell_lines[-n_rare_cell_lines:]

    val_es_cell_lines = random.sample(train_val_es_cell_lines, int(val_frac*len(train_val_es_cell_lines)))
    train_cell_lines = [x for x in train_val_es_cell_lines if x not in val_es_cell_lines]


    for i in range(number_of_folds):
        folds[i]['train'] = list(synergy_df[synergy_df['Cell_line'].isin(train_cell_lines)].index)
        folds[i]['val_es'] = list(synergy_df[synergy_df['Cell_line'].isin(val_es_cell_lines)].index)
        folds[i]['test'] = list(synergy_df[synergy_df['Cell_line'].isin(test_cell_lines)].index)

        neg_folds[i]['train'] = list(non_synergy_df[non_synergy_df['Cell_line'].isin(train_cell_lines)].index)
        neg_folds[i]['val_es'] = list(non_synergy_df[non_synergy_df['Cell_line'].isin(val_es_cell_lines)].index)
        neg_folds[i]['test'] = list(non_synergy_df[non_synergy_df['Cell_line'].isin(test_cell_lines)].index)

    return folds, neg_folds, non_synergy_df

def create_test_val_train_cross_val_folds(synergy_df, init_non_synergy_df, cross_val_type, number_of_folds, \
                                          neg_fact, val_frac, neg_sampling_type, n_rare_cell_lines=10):
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

    if cross_val_type == 'leave_comb':
        #split the positive edges
        folds, drug_pair_folds = leave_comb_cross_val(synergy_df, number_of_folds, val_frac)
        #generate/get the negative edges
        non_synergy_df=pd.DataFrame()
        for cell_line in cell_lines:
            if neg_sampling_type == 'degree_based':
                #generate more than needed neg samples to make sure that after downstream processing pos:neg is still 1:1
                neg_df = cell_line_based_negative_sampling_degree_based(cell_line, synergy_df, neg_fact+0.5)
                non_synergy_df = pd.concat([non_synergy_df, neg_df], axis=0)
                print('neg cell line: ', cell_line)
            elif neg_sampling_type == 'semi_random':
                print('neg cell line: ', cell_line)
                neg_df = cell_line_based_negative_sampling_semi_random(cell_line, synergy_df, neg_fact+0.5)
                non_synergy_df = pd.concat([non_synergy_df, neg_df], axis=0)

            elif neg_sampling_type == 'no':
                non_synergy_df = init_non_synergy_df
                break
        non_synergy_df.reset_index(drop=True, inplace= True)

        print('hello: ',len(synergy_df), len(non_synergy_df))
        # split the negative edges
        # common_pairs = set(zip(non_synergy_df['Drug1_pubchem_cid'],non_synergy_df['Drug2_pubchem_cid'])).\
        #     intersection(set(zip(synergy_df['Drug1_pubchem_cid'],synergy_df['Drug2_pubchem_cid'])))

        neg_folds, neg_drug_pair_folds = leave_comb_cross_val(non_synergy_df, number_of_folds, val_frac, folds)


    elif cross_val_type == 'leave_cell_line':
        #split the positive edges
        folds, cell_lines_per_fold = leave_cell_line_cross_val(synergy_df, number_of_folds, val_frac)

        #generate/get the negative edges
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

        # split the negative edges
        neg_folds,_ = leave_cell_line_cross_val(non_synergy_df, number_of_folds, val_frac, \
                                                cell_lines_per_fold=cell_lines_per_fold)



    elif cross_val_type == 'leave_comb_cell_line':
        #split the positive edges
        folds = leave_comb_cell_line_cross_val(synergy_df, number_of_folds, val_frac)

        #generate/get the negative edges
        non_synergy_df=pd.DataFrame()
        for cell_line in cell_lines:
            if neg_sampling_type == 'degree_based':
                neg_df = cell_line_based_negative_sampling_degree_based(cell_line, synergy_df, neg_fact+0.5)
                non_synergy_df = pd.concat([non_synergy_df, neg_df], axis=0)
                print('neg cell line: ', cell_line)
            elif neg_sampling_type == 'semi_random':
                neg_df = cell_line_based_negative_sampling_semi_random(cell_line, synergy_df, neg_fact+0.5)
                non_synergy_df = pd.concat([non_synergy_df, neg_df], axis=0)
                print('neg cell line: ', cell_line)
            elif neg_sampling_type == 'no':
                non_synergy_df = init_non_synergy_df
                break
        non_synergy_df.reset_index(drop=True, inplace= True)

        # split the negative edges
        neg_folds = leave_comb_cell_line_cross_val(non_synergy_df, number_of_folds, val_frac,
                                                folds)

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

    elif cross_val_type == 'leave_drug':
        # split the positive edges
        folds = leave_drug_cross_val(synergy_df, number_of_folds, val_frac)
        #generate/get the negative edges
        if neg_sampling_type == 'degree_based':
            neg_folds, non_synergy_df = drug_based_negative_sampling_degree_based(folds, synergy_df, neg_fact, number_of_folds)
        elif neg_sampling_type == 'semi_random':
            neg_folds, non_synergy_df = drug_based_negative_sampling_semi_random(folds, synergy_df, neg_fact, number_of_folds)
        elif neg_sampling_type == 'no':
            non_synergy_df = init_non_synergy_df
            neg_folds = leave_drug_cross_val(init_non_synergy_df, number_of_folds, val_frac)

    elif cross_val_type=='test_rare_cell_lines':
        folds, neg_folds, non_synergy_df = create_rare_cell_line_split(synergy_df, init_non_synergy_df, n_rare_cell_lines, number_of_folds, neg_fact,
                                        val_frac, neg_sampling_type)

    assert is_train_test_val_val_es_non_overlapping(folds, synergy_df) == True, 'problem in train-test-val split of positive edges'
    assert is_train_test_val_val_es_non_overlapping(neg_folds,
                                             non_synergy_df) == True, 'problem in train-test-val split of negative edges'


    #make sure that the pos neg ration is close to neg_fact
    for fold in range(number_of_folds):
        for split in ['train','test','val_es']:
            print('actual neg fact:', float(len(folds[fold][split]))/len(neg_folds[fold][split]))
            assert (float(len(folds[fold][split]))/len(neg_folds[fold][split])==neg_fact), 'problem in neg_fact'
    return folds, neg_folds, non_synergy_df


