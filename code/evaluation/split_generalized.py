#!/usr/bin/env python
import numpy as np
import random
import pandas as pd
import os
import copy
import pickle
import math

def col_filtering(values, col_names, df, multi):
    '''
    Given a list of tuples and a list of column names, filter the given df such that the values (tuples) in col_names are in the list of tuples.
    :return: the filtered df
    '''
    if multi: #consider columns together that is try to filter according to tuples that combines column values.
        return df[df[col_names].apply(tuple, axis=1).isin(values)]
    else: #consider columns separately.
        new_df = copy.deepcopy(df)
        for col_name in col_names:
            new_df = new_df[new_df[col_name].isin(values)]
        return new_df


def split_list(d_list, n_folds):
    '''
    :param d_list: list of tuples where each drug pairs or list of drugs or list of cell lines.
    :param n_folds: In how many folds to split the drug_combs list
    :return: a dict where key=fold number, value=list of drug_pairs in that fold.
    '''
    #split in n_folds
    random.shuffle(d_list)

    split_size= int(len(d_list) / n_folds)
    folds= {i: d_list[split_size * i: split_size * (i + 1)] for i in range(n_folds-1)}
    folds[n_folds-1] = d_list[split_size*(n_folds-1):]
    return folds


def verify_split(train_df, test_df, split_type):
    '''

    :param df: dataframe
    :param train_df: pandas dataframe
    :param val_df: pandas dataframe
    :return: Given a df with sample data (e.g., synergy triplets), verify that the train and test
    data have been created properly according to the split type.
    '''

    if split_type=='random':
        test_triplets = set(zip(test_df['source'],test_df['target'],test_df['edge_type']))
        train_triplets = set(zip(train_df['source'],train_df['target'],train_df['edge_type']))
        n_common = len(test_triplets.intersection(train_triplets))
        assert n_common == 0, print(f'error in {split_type} split')


    if split_type=='leave_comb':
        test_edges = set(zip(test_df['source'],test_df['target']))
        train_edges = set(zip(train_df['source'],train_df['target']))
        n_common = len(test_edges.intersection(train_edges))
        assert n_common == 0, print(f'error in {split_type} split')


    if split_type=='leave_drug':
        test_nodes = set(test_df['source']).union(set(test_df['target']))
        train_nodes = set(train_df['source']).union(set(train_df['target']))
        n_common = len(test_nodes.intersection(train_nodes))
        assert n_common == 0, print(f'error in {split_type} split')


    if split_type=='leave_cell_line':
        test_edge_type = set(test_df['edge_type'])
        train_edge_type = set(train_df['edge_type'])
        n_common = len(test_edge_type.intersection(train_edge_type))
        assert n_common == 0, print(f'error in {split_type} split')


def get_random_train_test(df, test_frac=0.2):

    test_size = int(len(df)*test_frac)
    test_df = df.sample(n=test_size)
    # prepare train and val split
    train_df = df[~(df['ID'].isin(list(test_df['ID'])))]
    return train_df, test_df


def get_random_n_split(df, n_folds):
    '''
    df: columns = ['source', 'target']
    function:  If edge with a certain edge_type appear in train, the same edge with the same edge_type will not
    appear in test. However, the same edge (i.e., same source and target) from another edge type may appear in test.
    '''
    # prepare test split.
    indices = list(range(len(df)))
    fold_size = int(len(indices)/n_folds)

    remaining_idx = copy.deepcopy(indices)
    train_idx = {}
    val_idx = {}
    for i in range(n_folds):
        val_idx[i] = random.sample(remaining_idx, fold_size)
        train_idx[i] = list(set(indices).difference(set(val_idx[i])))
        remaining_idx = list(set(remaining_idx).difference(set(val_idx[i])))
    return train_idx, val_idx

def get_edge_split_train_test(df, test_frac, tolerance=0.05):
    '''
        Input: synergy_df = a dataframe with atleast two columns ['source','target'].

        Function: edge appearing in train (irrespective of the edge type) will not appear in test.
    '''
    edges = list(set(zip(df['source'], df['target'])))  # list of tuples
    count=0
    min_diff=1
    #prepare test split.
    while True:
        count+=1
        test_size= int(len(edges)*test_frac)
        test_edges = random.sample(edges, test_size)  # sample 1/n_folds of drug_combs to put in test_data
        test_df = col_filtering(list(test_edges), ['source', 'target'], df, multi=True)

        cur_test_frac = len(test_df) / len(df)
        diff = abs(cur_test_frac-test_frac)

        if min_diff > diff:
            min_diff = diff
            best_test_df = test_df

        if (diff> tolerance) and (count<500):
            print('test is not of right size')
            continue
        else:
            break
    train_df = df[~(df['ID'].isin(list(best_test_df['ID'])))]

    return train_df, best_test_df

def get_edge_n_split(df, n_folds):
    '''
        Input: synergy_df = a dataframe with atleast two columns ['source','target'].

        Function: edge appearing in train (irrespective of the edge type) will not appear in test.
    '''
    edges = list(set(zip(df['source'], df['target'])))  # list of tuples

    #prepare train and val split
    edge_folds = split_list(edges, n_folds)
    df_split = {i: col_filtering(edge_folds[i], ['source', 'target'], df, multi=True) for i in range(len(edge_folds))}

    train_idx = {}
    val_idx = {}
    for i in range(n_folds):
        val_idx[i] = list(df_split[i]['ID'])
        train_idx[i] = list(set(df['ID']).difference(set(val_idx[i])))
    return train_idx, val_idx


def get_node_split_train_test(df, test_frac, tolerance=0.05):

    '''
    Function: Split the nodes appear in source or target into train test. Then split the samples (e.g., triplets)
    training samples only contain training nodes and the same for test samples.
    '''
    print('total triplets: ', len(df))
    nodes = list(set(df['source']).union(set(df['target']))) # list of tuples
    count=0
    min_diff=1
    while True:
        count+=1
        test_size= int(len(nodes)*test_frac)
        test_nodes = random.sample(nodes, test_size)  # sample 1/n_folds of drug_combs to put in test_data
        test_df = col_filtering(test_nodes, ['source', 'target'], df, multi=False)

        cur_test_frac = len(test_df) / len(df)
        diff = abs(cur_test_frac - test_frac)
        if min_diff > diff:
            min_diff = diff
            best_test_df = test_df
            best_test_nodes = test_nodes
        if (diff < tolerance) or (count > 500):
            break
        else: #try again
            print('try: ', count)


    train_df = df[~(df['source'].isin(best_test_nodes)| df['target'].isin(best_test_nodes))]
    return train_df, best_test_df


def get_node_n_split(df, n_folds):

    '''
    Function: Split the nodes appear in source or target into train test. Then split the samples (e.g., triplets)
    training samples only contain training nodes and the same for test samples.
    '''
    print('total triplets: ', len(df))

    nodes = list(set(df['source']).union(set(df['target']))) # list of tuples

    # now create 5 folds from the remaining drug and use each fold as val and the rest as train
    node_folds = split_list(nodes, n_folds)
    df_split = {i: col_filtering(node_folds[i], ['source', 'target'], df, multi=False) for i in range(len(node_folds))}
    #catch: we will assign fold to triplets. It's expected that the triplets containing drugs from both
    # val and training set will not be used in training or validation. However, concatenate
    # triplets from the above 5 splits/folds will reduce the data unnecessarily. Let say, fold1 is validation fold.
    #In this case, any triplets containing drug from fold 2 and 3 should be part of training.
    train_idx = {}
    val_idx = {}
    for i in range(n_folds):
        val_idx[i] = list(df_split[i]['ID'])
        train_df = df[~(df['source'].isin(node_folds[i]) | df['target'].isin(node_folds[i]))]
        train_idx[i] = list(train_df['ID'])

    return train_idx, val_idx


def get_edge_type_split_train_test(df, test_frac, tolerance=0.05):

    '''
    :param df:
    :param n_folds:
    :return:
    '''

    print('total triplets: ', len(df))
    edge_types = list(df['edge_type'].unique())# list of strings

    count=0
    min_diff=1
    while True:
        count+=1
        test_size = int(len(edge_types)*test_frac)
        test_edge_types = random.sample(edge_types, test_size)  # sample 1/n_folds of drug_combs to put in test_data
        test_df = df[df['edge_type'].isin(test_edge_types)]

        cur_test_frac = len(test_df) / len(df)
        diff = abs(cur_test_frac - test_frac)
        if min_diff > diff:
            min_diff = diff
            best_test_df = test_df
        if (diff < tolerance) or (count > 500):
           break
        else:
            print('try: ', count)


    train_df = df[~(df['ID'].isin(list(best_test_df['ID'])))]

    return train_df, best_test_df


def get_edge_type_n_split(df, n_folds):

    '''
    :param df:
    :param n_folds:


    :return:
    '''

    print('total triplets: ', len(df))
    edge_types = list(df['edge_type'].unique())# list of strings
    # now create 5 folds from the remaining drug and use each fold as val and the rest as train
    edge_type_folds = split_list(edge_types, n_folds)

    df_split = {i: col_filtering(edge_type_folds[i], ['edge_type'], df, multi=False) for i in range(len(edge_type_folds))}
    train_idx = {}
    val_idx = {}
    for i in range(n_folds):
        val_idx[i] = list(df_split[i]['ID'])
        train_idx[i] = list(set(df['ID']).difference(set(val_idx[i])))

    return train_idx, val_idx



def generalize_data(df):
    '''map drug_pids and cell_line_names to numerical index, here we consider drugs and cell lines
       for which the user defined required features are available, i.e., if feature='must', then only
       the drugs and cell lines for which we have all the features available appear here.'''

    #if the data contained in synergy_df does not change across runs, then the serial for each triplet in synergy_df and
    #the numerical index of drug and cell_lines should also be the same. Hence the sorting.
    df = df.sort_values(by=['drug_1_pid', 'drug_2_pid', 'cell_line_name'])

    drug_pids = list(set(df['drug_1_pid']).union(set(df['drug_2_pid'])))
    drug_pids.sort()

    cell_line_names = list(set(df['cell_line_name']))
    cell_line_names.sort()

    drug_2_idx = {pid: idx for (idx, pid) in enumerate(drug_pids)}
    cell_line_2_idx = {name: idx for (idx, name) in enumerate(cell_line_names)}

    df['source'] = df['drug_1_pid'].astype(str).apply(lambda x: drug_2_idx[x])
    df['target'] = df['drug_2_pid'].astype(str).apply(lambda x: drug_2_idx[x])
    df['edge_type'] = df['cell_line_name'].astype(str).apply(lambda x: cell_line_2_idx[x])

    return df, drug_2_idx, cell_line_2_idx

def wrapper_train_test(df, split_type, test_frac, spec_dir, force_run=True):
    '''Rename column names to more generalized ones. Also, convert drug and cell line ids to numerical ids compatible with models.'''

    split_type_2_function_map = {'random': get_random_train_test, 'leave_comb': get_edge_split_train_test,
                      'leave_drug':get_node_split_train_test, 'leave_cell_line':get_edge_type_split_train_test}

    test_file = f'{spec_dir}/test.tsv'
    train_file = f'{spec_dir}/train.tsv'

    all_triplets_file = f'{spec_dir}/all.tsv'
    drug_idx_file = f'{spec_dir}/drug_2_idx.tsv'
    cell_idx_file = f'{spec_dir}/cell_line_2_idx.tsv'

    summary = f'{spec_dir}/train_test_summary.txt'

    if (not os.path.exists(test_file)) or (force_run):
        df, drug_2_idx, cell_line_2_idx = generalize_data(df)

        df['ID'] = list(range(len(df)))
        print('Creating train test folds')
        train_df, test_df = split_type_2_function_map[split_type](df, test_frac)
        #save in file
        test_df.drop(columns='ID', inplace=True)
        train_df.drop(columns='ID', inplace=True)
        df.drop(columns='ID', inplace=True)
        drug_2_idx_df = pd.DataFrame({'pid': list(drug_2_idx.keys()), 'idx': list(drug_2_idx.values())})
        cell_2_idx_df = pd.DataFrame({'cell_line_name': list(cell_line_2_idx.keys()), 'idx': list(cell_line_2_idx.values())})


        os.makedirs(spec_dir, exist_ok=True)
        test_df.to_csv(test_file, sep='\t')
        train_df.to_csv(train_file, sep='\t')
        df.to_csv(all_triplets_file, sep='\t')
        drug_2_idx_df.to_csv(drug_idx_file, sep='\t')
        cell_2_idx_df.to_csv(cell_idx_file, sep='\t')


    else:
        print(f'Loading train test data from {spec_dir}')
        test_df = pd.read_csv(test_file, sep='\t', dtype={'drug_1_pid':str, 'drug_2_pid': str})
        train_df = pd.read_csv(train_file, sep='\t', dtype={'drug_1_pid':str, 'drug_2_pid': str})
        df = pd.read_csv(all_triplets_file, sep='\t', dtype={'drug_1_pid':str, 'drug_2_pid': str})

        drug_2_idx_df = pd.read_csv(drug_idx_file, dtype={'pid':str}, sep='\t')
        drug_2_idx = dict(zip(drug_2_idx_df['pid'], drug_2_idx_df['idx']))

        cell_line_2_idx_df = pd.read_csv(cell_idx_file, sep='\t')
        cell_line_2_idx = dict(zip(cell_line_2_idx_df['cell_line_name'],cell_line_2_idx_df['idx']))

    verify_split(train_df, test_df, split_type)

    test_drugs = set(test_df['source']).union(set(test_df['target']))
    test_cell_lines = set(test_df['edge_type'])
    print(f'TEST #triplets: {len(test_df)} \n #drugs: {len(test_drugs)}'
          f' \n #cell lines: {len(test_cell_lines)}')

    with open(summary, 'w') as file:
        file.write(f'TEST #triplets: {len(test_df)} \n #drugs: {len(test_drugs)}'
          f' \n #cell lines: {len(test_cell_lines)}')
    file.close()
    return train_df, test_df, drug_2_idx, cell_line_2_idx

def wrapper_nfold_split(df, split_type, n_folds, spec_dir, force_run=True):

    df['ID'] = list(range(len(df)))
    split_type_2_function_map = {'random': get_random_n_split, 'leave_comb': get_edge_n_split,
                      'leave_drug':get_node_n_split, 'leave_cell_line':get_edge_type_n_split}

    val_file = f'{spec_dir}/val_nfolds.pkl'
    train_file = f'{spec_dir}/train_nfolds.pkl'
    summary = f'{spec_dir}/n_fold_summary.txt'

    if (not os.path.exists(train_file)) or (force_run):
        print('Creating train val folds')
        train_idx, val_idx = split_type_2_function_map[split_type](df, n_folds)
        os.makedirs(spec_dir, exist_ok=True)
        with open(val_file, 'wb') as file:
            pickle.dump(val_idx, file)
        with open(train_file, 'wb') as file:
            pickle.dump(train_idx, file)


    else:
        print(f'Loading train val folds from {spec_dir}')
        with open(val_file, 'rb') as file:
            val_idx = pickle.load(file)
        with open(train_file, 'rb') as file:
            train_idx = pickle.load(file)

    #verify splits
    for i in range(n_folds):
        foldwise_train_df = df[df['ID'].isin(train_idx[i])]
        foldwise_val_df = df[df['ID'].isin(val_idx[i])]
        verify_split(foldwise_train_df,foldwise_val_df, split_type)

    file = open(summary, 'w')
    for i in range(n_folds):
        print(f'fold {i} # TRAIN triplets:  {len(train_idx[i])}')
        print(f'fold {i} # VAL triplets:  {len(val_idx[i])}')
        file.write(f'fold {i} # TRAIN triplets:  {len(train_idx[i])}'
                   f'\n fold {i} # VAL triplets:  {len(val_idx[i])} \n')
    file.close()
    return train_idx, val_idx



if __name__=='__main__':
    if 'snakemake' in globals():
        synergy_file = snakemake.input[0]
        split_type = snakemake.params[0]
        n_folds = snakemake.params[1]
        out_dir = snakemake.params[2]
    else:
        synergy_file='/home/grads/tasnina/Projects/Plug and Play/inputs/synergy/synergy_scores.tsv'
        split_type = 'leave_cell_line'
        n_folds = 5
        out_dir = "/home/grads/tasnina/Projects/Plug and Play/evaluation/splits"

    synergy_df = pd.read_csv(synergy_file, sep='\t',
                             dtype={'drug_1_pid': str, 'drug_2_pid': str, 'cell_line_name': str})
    synergy_df = synergy_df.reset_index(drop=True)  # index the pandas dataframe from 1 to n
    synergy_df.index.name = 'ID'
    synergy_df = synergy_df.reset_index()

    wrapper_train_test(synergy_df, split_type, n_folds, out_dir)
