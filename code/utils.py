import yaml
import pandas as pd
import numpy as np
from itertools import combinations
from itertools import product

def load_yaml_file(filepath):
    with open(filepath, 'r') as file:
        data = yaml.safe_load(file)
    return data



#********************************** SYNERGY TRIPLETS ***********************************
def feature_based_filtering(synergy_df, dfeat_dict, cfeat_dict, feature='must'):
    '''
    If none of the features are optional, then we need to filter out the triplets such that only drugs and cell lines
    with all feature information available are in the final synergy triplets.
    '''
    drug_pids = set(synergy_df['drug_1_pid']).union(set(synergy_df['drug_2_pid']))
    cell_line_names = set(synergy_df['cell_line_name'])
    print(f'Before filtering: \n#of triplets : {len(synergy_df)},\n#drugs {len(drug_pids)},'
          f' \n#cell lines {len(cell_line_names)}')

    if feature=='must':
        # find drugs with all features available
        for feat_name in dfeat_dict:
            if isinstance(dfeat_dict[feat_name],pd.DataFrame):
                drugs = set(dfeat_dict[feat_name]['pid'])
            elif isinstance(dfeat_dict[feat_name],dict):
                drugs = set(dfeat_dict[feat_name].keys())
            drug_pids = drug_pids.intersection(drugs)

        # find cell lines with all features available
        for feat_name in cfeat_dict:
            cells = set(cfeat_dict[feat_name]['cell_line_name'])
            cell_line_names = cell_line_names.intersection(cells)

            print(f'filtering for {feat_name}:{len(cell_line_names)}')

        #filter synergy triplets
        synergy_df = synergy_df[(synergy_df['drug_1_pid'].isin(drug_pids)) & (synergy_df['drug_2_pid'].isin(drug_pids))
                                & (synergy_df['cell_line_name'].isin(cell_line_names)) ]

    # n_after_feat_filt =len(synergy_df)
    # if k>0: #keep only top k cell lines having the most synergy triplets.
    #     top_cell_lines = synergy_df['cell_line_name'].value_counts().nlargest(k).index
    #     print('top cell lines:' , top_cell_lines)
    #     synergy_df = synergy_df[synergy_df['cell_line_name'].isin(top_cell_lines)]
    #
    #     print(f'keeping top {k} cell lines, retrieved frac:{len(synergy_df)/n_after_feat_filt}')
    #assert that there is no duplicate triplets in synergy_df
    triplets = list(zip(synergy_df['drug_1_pid'],synergy_df['drug_2_pid'],synergy_df['cell_line_name']))
    assert len(set(triplets))==len(triplets), print('still some duplicates remaining')

    print_synergy_stat(synergy_df)
    return synergy_df


def print_synergy_stat(synergy_df):
    drug_pids = set(synergy_df['drug_1_pid']).union(set(synergy_df['drug_2_pid']))
    cell_line_names = set(synergy_df['cell_line_name'])
    print(f'After filtering: \n#of triplets : {len(synergy_df)},\n#drugs {len(drug_pids)},'
          f' \n#cell lines {len(cell_line_names)}')

def abundance_based_filtering(synergy_df, min_frac=0.01):
    # Initialize the filtered_df as empty
    filtered_df = pd.DataFrame()

    # Sort cell lines by their count in descending order
    cell_line_counts = synergy_df['cell_line_name'].value_counts()

    for cell_line, count in cell_line_counts.items():
        # Calculate the potential new total rows if this cell line is added
        new_total_rows = len(filtered_df) + count
        new_fraction = count / new_total_rows

        # If adding this cell line meets the threshold, add it to filtered_df
        if new_fraction >= min_frac:
            filtered_df = pd.concat([filtered_df, synergy_df[synergy_df['cell_line_name'] == cell_line]])

    print('remaining #triplets: ', len(filtered_df))
    print('frac triplets retrieved: ', len(filtered_df)/len(synergy_df))
    print('#cell lines retrieved: ', len(filtered_df['cell_line_name'].unique()))
    print('cell lines retrieved: ', filtered_df['cell_line_name'].unique())
    return filtered_df
#
# def filter_cell_lines(synergy_df, min_frac=0.01):
#     total_rows = len(synergy_df)
#     # cell_line_counts = synergy_df['cell_line_name'].value_counts()
#     fraction_per_cell_line = synergy_df['cell_line_name'].value_counts() / total_rows
#
#     filtered_cell_lines = fraction_per_cell_line[fraction_per_cell_line >= min_frac].index
#     filtered_df = synergy_df[synergy_df['cell_line_name'].isin(filtered_cell_lines)]
#
#     print('remaining #triplets: ', len(filtered_df))
#     print('frac triplets retrieved: ', len(filtered_df)/len(synergy_df))
#     print('#cell lines retrieved: ', len(filtered_df['cell_line_name'].unique()))
#     print('cell lines retrieved: ', filtered_df['cell_line_name'].unique())
#
#
#     return filtered_df

def compute_feature_combination(feature_info, max_size=None):
    '''
    Computes valid combinations of features for training a model based on the 'use' values.
    :param feature_info: a list of dictionaries, where each dictionary has 3 keys: 'name', 'preprocess', and 'use'.
        'name' is a string representing the feature name.
        'use' is a list of boolean values indicating whether a feature can be used.
    :return: a list of lists, where each sublist represents a valid combination of feature names to use.
    '''
    # Filter and collect feature names that have at least one 'True' in their 'use' list
    available_features = [feature['name'] for feature in feature_info if True in feature['use']]
    must_features = [feature['name'] for feature in feature_info if False not in feature['use']]
    optional_features = [feature for feature in available_features if feature not in must_features]
    # Initialize an empty list to store all non-empty combinations
    all_combinations = []
    # Generate all possible non-empty combinations of available features
    if max_size is None: # in this case generate combination of all possible sizes
        max_size = len(optional_features)
    else:
        max_size = (max_size-len(must_features))
        assert max_size>=0, print('#of must present features exceeds allowed #of features.')
    for i in range(max_size+1):
        all_combinations.extend(combinations(optional_features, i))
    # Add the must have features. Convert tuples in the list to lists.
    feat_combinations = [list(combo)+must_features for combo in all_combinations]
    #filter empty feature combs
    feat_combinations = [feat_comb for feat_comb in feat_combinations if len(feat_comb)>0]
    #make sure that at least one feature comb is being chosen.
    assert len(feat_combinations)>0, print('ERROR: no feature is selected to use.')
    return feat_combinations

def find_drug_cell_feat_combs(drug_feat_combs, cell_feat_combs):
    '''
        Generates all possible combinations of drug features with cell features.
    :param drug_feat_combs: a list of lists, each sublist containing names of drug features.
    :param cell_feat_combs: a list of lists, each sublist containing names of cell features.
    :return: a list of tuples, where each tuple contains a list from drug_feat_combs and a list from cell_feat_combs.

    '''
    return list(product(drug_feat_combs, cell_feat_combs))

def check_config_compeleteness(entity, features_info, models_info):
    # make sure that whatever features you preprocess they are checked out, as in, if any feature
    # needs encoder and the encoder is not mentioned then raise an error message.
    for feat_info in features_info:
        if feat_info['encoder']:
            feat_needing_encoder = feat_info['name']
            flag = 0
            for model_info in models_info:
                encoders = model_info.get(f'{entity}_encoder', [])
                for encoder in encoders:
                    if encoder['feat'] == feat_needing_encoder:
                        flag = 1
                assert flag == 1, print(f'encoder information missing for feature:'
                                        f'{feat_needing_encoder}')

def generalize_data(df, col_name_map= {'drug_1_pid': 'source', 'drug_2_pid': 'target', 'cell_line_name': 'edge_type'}):

    # print(f'\n\nAfter filtering: \n#of triplets : {len(synergy_df)}\n#drugs {len(drug_pids)}'
    #       f' \n#cell lines {len(cell_line_names)}')
    #
    '''map drug_pids and cell_line_names to numerical index, here we consider drugs and cell lines
       for which the user defined nrequired features are available, i.e., if feature='must', then only
       the drugs and cell lines for which we have all the features available appera here.'''

    drug_pids = set(df['drug_1_pid']).union(set(df['drug_2_pid']))
    cell_line_names = set(df['cell_line_name'])

    drug_2_idx = {pid: idx for (idx, pid) in enumerate(drug_pids)}
    cell_line_2_idx = {name: idx for (idx, name) in enumerate(cell_line_names)}

    df = df.rename(columns=col_name_map)

    df['source'] = df['source'].astype(str).apply(lambda x: drug_2_idx[x])
    df['target'] = df['target'].astype(str).apply(lambda x: drug_2_idx[x])
    df['edge_type'] = df['edge_type'].astype(str).apply(lambda x: cell_line_2_idx[x])

    return df, drug_2_idx, cell_line_2_idx

def get_feat_prefix(params, select_drug_feat, select_cell_feat):
    drug_prefix = 'drug_' + '_'.join(select_drug_feat)

    cell_prefix = '_cell_' + '_'.join(select_cell_feat)

    other_prefix = f'_{params.feature}'

    file_prefix =  drug_prefix + cell_prefix + other_prefix

    return file_prefix

def create_file_prefix(params, select_drug_feat, select_cell_feat, model_name, split_type):
    dir_prefix = f"{params.out_dir}/k_{params.abundance}/{split_type}/{model_name}/"
    file_prefix = dir_prefix + get_feat_prefix(params, select_drug_feat, select_cell_feat)
    return file_prefix

# "/home/grads/tasnina/Projects/SynVerse/outputs//k_6/leave_drug/MLP/_drug_1_-_h_o_t_cell_1_-_h_o_t_must/"

def normalize(df, std1=None, means1=None, norm_type = 'std'):
    '''
    :param X: a numpy 2D array to normalize
    :param norm_type: how to normalize
    :return: normalized numpy array with same number of rows as X
    '''
    if norm_type=='std':
        if (std1 is None) | (means1 is None):
            means1 = df.mean().values
            std1 = df.std().values
            df = (df-means1)/std1
        else:
            df = (df - means1) / std1
        return df, means1, std1
