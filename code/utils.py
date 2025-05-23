import os
import sys

import torch
import subprocess
import yaml
import pandas as pd
from itertools import combinations
from itertools import product
import copy
def load_yaml_file(filepath):
    with open(filepath, 'r') as file:
        data = yaml.safe_load(file)
    return data


def get_available_gpu():
    # If CUDA isn’t available or there are no GPUs, fall back to CPU
    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        print("No GPUs available, using CPU.")
        return torch.device("cpu")

    # Query free/total memory for all GPUs
    result = subprocess.run(
        ['nvidia-smi',
         '--query-gpu=memory.free,memory.total',
         '--format=csv,noheader,nounits'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True
    )
    lines = result.stdout.decode('utf-8').strip().splitlines()

    # Build a list of free-memory ints
    free_mem = [int(line.split(',')[0]) for line in lines]

    # Select the GPU index that has the most free memory
    best_gpu = max(range(len(free_mem)), key=lambda i: free_mem[i])
    print(f"Using GPU {best_gpu} with {free_mem[best_gpu]} MB free memory.")

    return torch.device(f"cuda:{best_gpu}")

#********************************** SYNERGY TRIPLETS ***********************************
def sort_paired_cols(df, col1, col2, inplace, relation='greater'):
    '''
    Given a pandas dataframe,df and two column names, this function will return a
    dataframe where col1>col2.
    '''
    if relation=='greater':
        mask = df[col1] < df[col2]
    elif relation=='less':
        mask = df[col1] > df[col2]

    if inplace:
        df.loc[mask, [col1, col2]] = df.loc[mask, [col2, col1]].values
    else:
        df_new = copy.deepcopy(df)
        df_new.loc[mask, [col1, col2]] = df_new.loc[mask, [col2, col1]].values
        return df_new

def print_synergy_stat(synergy_df):
    drug_pids = set(synergy_df['drug_1_pid']).union(set(synergy_df['drug_2_pid']))
    drug_pairs = set(zip(synergy_df['drug_1_pid'], synergy_df['drug_2_pid']))

    cell_line_names = set(synergy_df['cell_line_name'])
    print(f'#of triplets : {len(synergy_df)},\n #unique drug pairs {len(drug_pairs)}\n#unique drugs {len(drug_pids)},'
          f' \n#unique cell lines {len(cell_line_names)}\n')

    print('minimum triplets in a cell line: ', list(synergy_df['cell_line_name'].value_counts())[-1])
    return drug_pairs, drug_pids, cell_line_names



def get_feature_comb_wrapper(dfeat_dict, cfeat_dict, max_drug_feat, min_drug_feat, max_cell_feat, min_cell_feat):
    def compute_feature_combination(feature_info, max_size=None, min_size=1):
        '''
        Computes valid combinations of features for training a model based on the 'use' values.
        :param feature_info: a list of dictionaries, where each dictionary has 3 keys: 'name', 'preprocess', and 'use'.
            'name' is a string representing the feature name.
            'use' is a list of boolean values indicating whether a feature can be used.
        :return: a list of lists, where each sublist represents a valid combination of feature names to use.
        '''
        # Filter and collect feature names that have at least one 'True' in their 'use' list
        available_features = [feature_name for feature_name in feature_info if
                              True in feature_info[feature_name]]  # feature with use=[True, False] or use=[True]
        must_features = [feature_name for feature_name in feature_info if
                         False not in feature_info[feature_name]]  # feature with use=[True]
        optional_features = [feature_name for feature_name in available_features if
                             feature_name not in must_features]  # feature with use=[True, False]
        # Initialize an empty list to store all non-empty combinations
        all_combinations = []
        # Generate all possible non-empty combinations of available features
        if max_size is None:  # in this case generate combination of all possible sizes
            max_size = len(optional_features)
        else:
            max_size = (max_size - len(must_features))
            assert max_size >= 0, print('Please increase max_drug_feat and/or max_cell_feat')
        for i in range(max_size + 1):
            all_combinations.extend(combinations(optional_features, i))
        # Add the must have features. Convert tuples in the list to lists.
        feat_combinations = [list(combo) + must_features for combo in all_combinations]
        # filter empty feature combs
        feat_combinations = [feat_comb for feat_comb in feat_combinations if len(feat_comb) > 0]
        # make sure that at least one feature comb is being chosen.
        assert len(feat_combinations) > 0, print('ERROR: no feature is selected to use.')

        # filter feature combs with < min_size
        feat_combinations = [feat_comb for feat_comb in feat_combinations if len(feat_comb) >= min_size]
        return feat_combinations

    def find_drug_cell_feat_combs(drug_feat_combs, cell_feat_combs):
        '''
            Generates all possible combinations of drug features with cell features.
        :param drug_feat_combs: a list of lists, each sublist containing names of drug features.
        :param cell_feat_combs: a list of lists, each sublist containing names of cell features.
        :return: a list of tuples, where each tuple contains a list from drug_feat_combs and a list from cell_feat_combs.

        '''
        return list(product(drug_feat_combs, cell_feat_combs))

    drug_feat_combs = compute_feature_combination(dfeat_dict['use'], max_size=max_drug_feat, min_size=min_drug_feat)
    cell_feat_combs = compute_feature_combination(cfeat_dict['use'], max_size=max_cell_feat, min_size=min_cell_feat)
    drug_cell_feat_combs = find_drug_cell_feat_combs(drug_feat_combs, cell_feat_combs)
    return drug_cell_feat_combs



def keep_selected_feat(feat_dict, selected_feat):
    fields = ['norm', 'preprocess', 'encoder', 'value', 'dim', 'use','compress', 'file']  # for each feature we can have these fields.
    select_feat_dict = {field: {} for field in fields}

    for field in fields:
        select_feat_dict[field] = {feat_name: feat_dict[field][feat_name] for feat_name in feat_dict[field]
                                   if feat_name in selected_feat}

    return select_feat_dict


def get_feat_prefix(dfeat_dict, cfeat_dict, mention_norm=False, mention_encoder=False, mention_preprocess=False,
                    mention_compress=False):
    def format_feat(feat_dict, feat_names):
        parts = []
        for feat_name in feat_names:
            details = [feat_name]  # Start with the feature name

            if mention_norm and feat_dict['norm'].get(feat_name):
                details.append(f"_{feat_dict['norm'][feat_name]}")
            if mention_preprocess and feat_dict['preprocess'].get(feat_name):
                details.append(f"_{feat_dict['preprocess'][feat_name]}")
            if mention_compress and feat_dict['compress'].get(feat_name):
                details.append(f"_comp_{feat_dict['compress'][feat_name]}")
            if mention_encoder and feat_dict['encoder'].get(feat_name):
                details.append(f"_{feat_dict['encoder'][feat_name]}")

            parts.append(''.join(details))

        return '_'.join(parts)

    dfeat_names = sorted(dfeat_dict['value'].keys())
    cfeat_names = sorted(cfeat_dict['value'].keys())

    dfeat_str = f"D_{format_feat(dfeat_dict, dfeat_names)}"
    cfeat_str = f"C_{format_feat(cfeat_dict, cfeat_names)}"

    return f"{dfeat_str}_{cfeat_str}".strip('_')


def create_file_prefix(params, select_dfeat_dict, select_cfeat_dict, split_type, split_feat_str='', run_no=None, seed=None):
    dir_prefix = f"{params.out_dir}/k_{params.abundance}_{params.score_name}/{split_type}/"
    if run_no is not None:
        dir_prefix=f'{dir_prefix}/run_{run_no}_{seed}/'
    feat_model_prefix = get_feat_prefix(select_dfeat_dict, select_cfeat_dict, mention_norm=True, mention_encoder=True, mention_preprocess=True,
                                        mention_compress=True)

    if feat_model_prefix == 'D_d1hot_C_c1hot': #when only 1hot feature is being used then we should keep track of which split it is running on as well
        dir_prefix = dir_prefix + '/One-hot-versions/' + split_feat_str + '/'
    file_prefix = dir_prefix + feat_model_prefix
    return file_prefix

def combine_hyperparams(model_info):
    '''
    model_info is a dict of dict like the following.
    decoder:
       name: 'MLP'
       hp_range: { 'lr': [ 1e-5, 1e-4 ], 'optimizer': [ 'Adam', 'SGD' ] , 'sgd_momentum': [ 0.0, 0.99 ] ,
                   'num_hid_layers': [ 1,3 ] , 'hid_0': [ 64, 2048 ] , 'hid_1': [ 64,2048 ] , 'hid_2': [ 64, 2048 ] ,
                   'in_dropout_rate': [ 0.0, 0.5 ] ,'hid_dropout_rate': [ 0.0, 0.5 ] }
       hp: {'hid_0': 1077, 'hid_dropout_rate': 0.2697798807963333, 'in_dropout_rate': 0.17927761184927016, 'lr': 0.00016381808734015385,
              'num_hid_layers': 2, 'optimizer': 'SGD', 'hid_1': 113, 'sgd_momentum': 0.21730082740135556}
     #
     drug_encoder:
       - name: 'GCN'
         hp_range: { 'batch_norm': [ True, False ], 'gnn_num_layers': [ 1,3 ], 'gnn_0': [ 64, 2048 ] ,'gnn_1': [ 64, 2048 ],'gnn_2': [ 64, 2048 ] ,
                     'ff_num_layers': [ 1,3 ], 'ff_0': [ 16, 2048 ] , 'ff_1': [ 16, 2048 ] , 'ff_2': [ 16, 2048 ] , 'gnn_dropout': [ 0.0,0.5 ] }
         hp: { 'gnn_num_layers': 2 ,'gnn_0': 512 ,'gnn_1': 128,
               'ff_num_layers': 1, 'ff_0': 128, 'gnn_dropout': 0.3 }
     cell_encoder:
       -name: null
    '''
    hp = {}
    decoder = model_info['decoder']
    hp.update(decoder.get('hp', {}))

    drug_encoders = model_info.get('drug_encoder',[])
    cell_encoders = model_info.get('cell_encoder',[])

    for drug_encoder in drug_encoders if drug_encoders is not None else [] :
        hp.update(drug_encoder.get('hp', {}))

    for cell_encoder in cell_encoders if cell_encoders is not None else [] :
        hp.update(cell_encoder.get('hp', {}))

    return hp


def get_select_model_info(model_info, select_dfeat_encoders, select_cfeat_encoders):
    decoder = model_info['decoder']
    drug_encoders = model_info.get('drug_encoder', [])
    cell_encoders = model_info.get('cell_encoder', [])

    select_model_info = {'decoder':decoder, 'drug_encoder':[], 'cell_encoder':[]}
    for drug_encoder in drug_encoders if drug_encoders is not None else [] :
        if drug_encoder['name'] in list(select_dfeat_encoders.values()):
            select_model_info['drug_encoder'].append(drug_encoder)
            print('keep drug encoder: ', drug_encoder['name'])

    for cell_encoder in cell_encoders if cell_encoders is not None else [] :
        if cell_encoder['name'] in list(select_cfeat_encoders.values()):
            select_model_info['cell_encoder'].append(cell_encoder)
            print('keep cell encoder: ', cell_encoder['name'])

    return select_model_info


def get_index_sorted_feature_matrix(dfeat_mtx_dict, drug_2_idx, cfeat_mtx_dict, cell_line_2_idx):
    # convert 'pid' and 'cell_line_name' to numerical index in the feature dictionaries.
    for feat_name in dfeat_mtx_dict:
        if isinstance(dfeat_mtx_dict[feat_name], pd.DataFrame):
            cur_dfeat = dfeat_mtx_dict[feat_name]
            cur_dfeat['idx'] = cur_dfeat['pid'].astype(str).apply(lambda x: drug_2_idx.get(x))
            cur_dfeat.drop_duplicates(subset=['pid'], inplace=True)
            cur_dfeat.dropna(subset=['idx'], inplace=True)
            cur_dfeat.set_index('idx', inplace=True)
            cur_dfeat.drop(axis=1, columns=['pid'], inplace=True)
            # sort drugs according to index
            cur_dfeat.sort_index(inplace=True)
            assert list(cur_dfeat.index) == list(range(len(cur_dfeat))), print('index not in order.')
            # save feature of drugs as numpy array
            dfeat_mtx_dict[feat_name] = cur_dfeat.values

        elif isinstance(dfeat_mtx_dict[feat_name], dict):
            dfeat_mtx_dict[feat_name] = {drug_2_idx[str(old_key)]: value for old_key, value in
                                            dfeat_mtx_dict[feat_name].items() if old_key in drug_2_idx}

    for feat_name in cfeat_mtx_dict:
        cur_cfeat = cfeat_mtx_dict[feat_name]
        cur_cfeat['idx'] = cur_cfeat['cell_line_name'].astype(str).apply(lambda x: cell_line_2_idx.get(x))
        cur_cfeat.dropna(subset=['idx'], inplace=True)
        cur_cfeat.set_index('idx', inplace=True)
        cur_cfeat.drop(axis=1, columns=['cell_line_name'], inplace=True)
        cur_cfeat.sort_index(inplace=True)
        assert list(cur_cfeat.index) == list(range(len(cur_cfeat))), print(
            'index not in order.')
        cfeat_mtx_dict[feat_name] = cur_cfeat.values

    return dfeat_mtx_dict, cfeat_mtx_dict



def extract_best_hyperparam(hyperparam_file):
    best_config = {}
    best_epochs = None

    with open(hyperparam_file, 'r') as f:
        for line in f:
            if line.startswith('best_config'):
                # Extract the dictionary string and convert it back to a dictionary
                best_config = eval(line.replace('best_config: ', '').strip())
            elif line.startswith('best_epochs'):
                # Extract the number and convert it to an integer
                best_epochs = int(line.replace('best_epochs: ', '').strip())

    print("Best Config:", best_config)
    print("Best Epochs:", best_epochs)
    return best_config, best_epochs





def remove_self_loop_from_splits(test_df, all_train_df, train_idx, val_idx):
    train_idx = train_idx[0]
    val_idx = val_idx[0]
    idx_with_self_loop_train = list(all_train_df[all_train_df['source'] == all_train_df['target']].index)
    idx_with_self_loop_test = list(test_df[test_df['source'] == test_df['target']].index)

    all_train_df_clean = all_train_df.drop(index=idx_with_self_loop_train).reset_index(drop=True)
    test_df_clean = test_df.drop(index=idx_with_self_loop_test).reset_index(drop=True)

    old_to_new_index_map = {
        old_idx: new_idx
        for new_idx, old_idx in enumerate(all_train_df.drop(index=idx_with_self_loop_train).index)
    }

    new_train_idx = [old_to_new_index_map[i] for i in train_idx if i in old_to_new_index_map]
    new_val_idx = [old_to_new_index_map[i] for i in val_idx if i in old_to_new_index_map]

    return test_df_clean, all_train_df_clean, {0:new_train_idx}, {0:new_val_idx}
#
# def check_remove_loop():
#     # Step 1: Create dummy all_train_df and test_df
#     all_train_df = pd.DataFrame({
#         'source': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'H', 'I'],
#         'target': ['B', 'C', 'C', 'D', 'E', 'F', 'H', 'K', 'D', 'I'],  # idx 4 and 8 are self-loops
#     })
#
#     test_df = pd.DataFrame({
#         'source': ['X', 'Y', 'Z'],
#         'target': ['X', 'B', 'Z'],  # idx 0 and 2 are self-loops
#     })
#
#     # Step 2: Define train_idx and val_idx using old DataFrame indices
#     train_idx = [0, 2, 4, 5, 7, 8]  # idx 4 and 8 should be dropped
#     val_idx = [1, 3, 6, 9]  # no dropped indices here
#
#     # Step 3: Run the function
#     clean_train_df, clean_test_df, new_train_idx, new_val_idx = remove_self_loop_from_splits(
#         all_train_df, test_df, train_idx, val_idx
#     )
#
#     # Step 4: Print results
#     print("Cleaned all_train_df:")
#     print(clean_train_df)
#
#     print("\nCleaned test_df:")
#     print(clean_test_df)
#
#     print("\nNew train_idx:", new_train_idx)
#     print("New val_idx:", new_val_idx)



# check_remove_loop()



