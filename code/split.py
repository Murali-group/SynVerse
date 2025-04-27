import pandas as pd
import os
import copy
import pickle
from graph_split import *

split_type_map = {'random': 'random', 'leave_comb': 'edge', 'leave_drug':'node', 'leave_cell_line': 'edge_type'}


def generalize_data(df, save_dir):
    '''map drug_pids and cell_line_names to numerical index, here we consider drugs and cell lines
       for which the user defined required features are available, i.e., if feature='must', then only
       the drugs and cell lines for which we have all the features available appear here.'''

    #if the data contained in synergy_df does not change across runs, then the serial for each triplet in synergy_df and
    #the numerical index of drug and cell_lines should also be the same. Hence the sorting.
    df = df.sort_values(by=['drug_1_pid', 'drug_2_pid', 'cell_line_name']).reset_index(drop=True)

    drug_pids = list(set(df['drug_1_pid']).union(set(df['drug_2_pid'])))
    drug_pids.sort()

    cell_line_names = list(set(df['cell_line_name']))
    cell_line_names.sort()

    drug_2_idx = {pid: idx for (idx, pid) in enumerate(drug_pids)}
    cell_line_2_idx = {name: idx for (idx, name) in enumerate(cell_line_names)}

    df['source'] = df['drug_1_pid'].astype(str).apply(lambda x: drug_2_idx[x])
    df['target'] = df['drug_2_pid'].astype(str).apply(lambda x: drug_2_idx[x])
    df['edge_type'] = df['cell_line_name'].astype(str).apply(lambda x: cell_line_2_idx[x])

    drug_2_idx_df = pd.DataFrame({'pid': list(drug_2_idx.keys()), 'idx': list(drug_2_idx.values())})
    cell_2_idx_df = pd.DataFrame({'cell_line_name': list(cell_line_2_idx.keys()), 'idx': list(cell_line_2_idx.values())})

    os.makedirs(save_dir, exist_ok=True)
    drug_2_idx_df.to_csv(f'{save_dir}/drug_2_idx.tsv', sep='\t')
    cell_2_idx_df.to_csv(f'{save_dir}/cell_line_2_idx.tsv', sep='\t')

    return df, drug_2_idx, cell_line_2_idx

# def wrapper_train_test(df, split_type, test_frac, spec_dir, seed=None):
#     '''Rename column names to more generalized ones. Also, convert drug and cell line ids to numerical ids compatible with models.'''
#     test_file = f'{spec_dir}/test.tsv'
#     train_file = f'{spec_dir}/train.tsv'
#     all_triplets_file = f'{spec_dir}/all.tsv'
#     summary = f'{spec_dir}/train_test_summary.txt'
#
#
#     print('Creating train test folds')
#     train_idx, test_idx = split_train_test(df, split_type_map[split_type], test_frac, seed)
#
#     train_df = df.iloc[train_idx]
#     test_df = df.iloc[test_idx]
#
#     os.makedirs(spec_dir, exist_ok=True)
#     test_df.to_csv(test_file, sep='\t')
#     train_df.to_csv(train_file, sep='\t')
#     df.to_csv(all_triplets_file, sep='\t')
#
#     test_drugs = set(test_df['source']).union(set(test_df['target']))
#     test_cell_lines = set(test_df['edge_type'])
#     print(f'TEST #triplets: {len(test_df)} \n #drugs: {len(test_drugs)}'
#           f' \n #cell lines: {len(test_cell_lines)}')
#
#     with open(summary, 'w') as file:
#         file.write(f'TEST #triplets: {len(test_df)} \n #drugs: {len(test_drugs)}'
#           f' \n #cell lines: {len(test_cell_lines)}')
#     file.close()
#     # return train_df, test_df, drug_2_idx, cell_line_2_idx
#     return train_df, test_df

def wrapper_test_train_val(df, split_type, test_frac, val_frac, spec_dir, seed=None, force_run=False):
    print(split_type)
    test_file = f'{spec_dir}/test.tsv'
    all_train_file = f'{spec_dir}/all_train.tsv'
    train_file =  f'{spec_dir}/train.pkl'
    val_file =  f'{spec_dir}/val.pkl'
    all_triplets_file = f'{spec_dir}/all.tsv'

    summary = f'{spec_dir}/train_test_val_summary.txt'

    os.makedirs(spec_dir, exist_ok=True)
    # print('Creating train test val folds')

    if (not (os.path.exists(test_file) and os.path.exists(train_file) and os.path.exists(val_file))) or force_run:
        print('Creating train test val folds')
        #splitting into train and test
        all_train_idx, test_idx = split_train_test(df, split_type_map[split_type], test_frac, seed)
        all_train_df = df.iloc[all_train_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx]

        #now split the train into val and train to be used in hyperparam tuning
        train_idx, val_idx = split_train_test(all_train_df, split_type_map[split_type], val_frac, seed)
        with open(train_file, 'wb') as file:
            pickle.dump(train_idx, file)
        with open(val_file, 'wb') as file:
            pickle.dump(val_idx, file)


        #save in file
        os.makedirs(spec_dir, exist_ok=True)
        test_df.to_csv(test_file, sep='\t')
        all_train_df.to_csv(all_train_file, sep='\t')
        df.to_csv(all_triplets_file, sep='\t')
    else:
        print(f'Loading train test val data from {spec_dir}')
        test_df = pd.read_csv(test_file, sep='\t', dtype={'drug_1_pid': str, 'drug_2_pid': str})
        all_train_df = pd.read_csv(all_train_file, sep='\t', dtype={'drug_1_pid': str, 'drug_2_pid': str})

        with open(val_file, 'rb') as file:
            val_idx = pickle.load(file)
        with open(train_file, 'rb') as file:
            train_idx = pickle.load(file)

    test_drugs = set(test_df['source']).union(set(test_df['target']))
    test_cell_lines = set(test_df['edge_type'])

    train_df = all_train_df.iloc[train_idx]
    train_drugs = set(train_df['source']).union(set(train_df['target']))
    train_cell_lines = set(train_df['edge_type'])

    val_df = all_train_df.iloc[val_idx]
    val_drugs = set(val_df['source']).union(set(val_df['target']))
    val_cell_lines = set(val_df['edge_type'])
    print(f'TEST #triplets: {len(test_df)} \n #drugs: {len(test_drugs)}'
          f' \n #cell lines: {len(test_cell_lines)}')
    print(f'TRAIN #triplets: {len(train_df)} \n #drugs: {len(train_drugs)}'
          f' \n #cell lines: {len(train_cell_lines)}')
    print(f'VAL #triplets: {len(val_df)} \n #drugs: {len(val_drugs)}'
          f' \n #cell lines: {len(val_cell_lines)}')

    with open(summary, 'w') as file:
        file.write(f'TEST #triplets: {len(test_df)} \n #drugs: {len(test_drugs)}'
                   f' \n #cell lines: {len(test_cell_lines)}\n\n'
                   f'TRAIN #triplets: {len(train_df)} \n #drugs: {len(train_drugs)}'
                   f' \n #cell lines: {len(train_cell_lines)}\n\n'
                   f' VAL #triplets: {len(val_df)} \n #drugs: {len(val_drugs)}'
                   f' \n #cell lines: {len(val_cell_lines)}')
    file.close()

    #TODO: convert the return type of train and val idx from a dictionary to list once we finalize on NOT using n_folds any more.
    # return test_df, all_train_df, {0:train_idx}, {0:val_idx}, drug_2_idx, cell_line_2_idx
    return test_df, all_train_df, {0:train_idx}, {0:val_idx}

