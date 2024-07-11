import yaml
import pandas as pd
import numpy as np
from itertools import combinations
from itertools import product
from torch_geometric import data as DATA
import torch


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
    print('Before feature based fitering: ')
    print_synergy_stat(synergy_df)


    if feature=='must':
        # find drugs with all features available
        for feat_name in dfeat_dict:
            if isinstance(dfeat_dict[feat_name],pd.DataFrame):
                drugs = set(dfeat_dict[feat_name]['pid'])
            elif isinstance(dfeat_dict[feat_name],dict):
                drugs = set(dfeat_dict[feat_name].keys())
            drug_pids = drug_pids.intersection(drugs)
            print(f'filtering for {feat_name}')


        # find cell lines with all features available
        for feat_name in cfeat_dict:
            cells = set(cfeat_dict[feat_name]['cell_line_name'])
            cell_line_names = cell_line_names.intersection(cells)

            print(f'filtering for {feat_name}')

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

    print('After feature based filtering: ')
    print_synergy_stat(synergy_df)
    return synergy_df


def print_synergy_stat(synergy_df):
    drug_pids = set(synergy_df['drug_1_pid']).union(set(synergy_df['drug_2_pid']))
    cell_line_names = set(synergy_df['cell_line_name'])
    print(f'\n#of triplets : {len(synergy_df)},\n#drugs {len(drug_pids)},'
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

    print('After abundance based filtering: ')
    print('frac triplets retrieved: ', len(filtered_df)/len(synergy_df))
    print_synergy_stat(filtered_df)

    return filtered_df

def get_feature_comb_wrapper(dfeat_names, dfeat_dict, cfeat_names, cfeat_dict,
                             use_feat=None, max_feat=None):
    # if kwargs.get('feat') is not None:
    if use_feat is not None:
        use_feat = use_feat.split(' ')
        for feat_name in dfeat_names:
            if feat_name in use_feat:
                dfeat_dict['use'][feat_name] = [True]
            else:
                dfeat_dict['use'][feat_name] = [False]
        for feat_name in cfeat_names:
            if feat_name in use_feat:
                cfeat_dict['use'][feat_name] = [True]
            else:
                cfeat_dict['use'][feat_name] = [False]
        max_feat=None

    drug_feat_combs = compute_feature_combination(dfeat_dict['use'], max_feat)
    cell_feat_combs = compute_feature_combination(cfeat_dict['use'], max_feat)
    drug_cell_feat_combs = find_drug_cell_feat_combs(drug_feat_combs, cell_feat_combs)
    return drug_cell_feat_combs
def compute_feature_combination(feature_info, max_size=None):
    '''
    Computes valid combinations of features for training a model based on the 'use' values.
    :param feature_info: a list of dictionaries, where each dictionary has 3 keys: 'name', 'preprocess', and 'use'.
        'name' is a string representing the feature name.
        'use' is a list of boolean values indicating whether a feature can be used.
    :return: a list of lists, where each sublist represents a valid combination of feature names to use.
    '''
    # Filter and collect feature names that have at least one 'True' in their 'use' list
    available_features = [feature_name for feature_name in feature_info if True in feature_info[feature_name]]
    must_features = [feature_name for feature_name in feature_info if False not in feature_info[feature_name]]
    optional_features = [feature_name for feature_name in available_features if feature_name not in must_features]
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


def keep_selected_feat(feat_dict, selected_feat):
    fields = ['norm', 'preprocess', 'encoder', 'mtx', 'dim', 'use']  # for each feature we can have these fields.
    select_feat_dict = {field: {} for field in fields}

    for field in fields:
        select_feat_dict[field] = {feat_name: feat_dict[field][feat_name] for feat_name in feat_dict[field]
                                   if feat_name in selected_feat}

    return select_feat_dict


# def check_config_compeleteness(entity, features_info, models_info):
#     # make sure that whatever features you preprocess they are checked out, as in, if any feature
#     # needs encoder and the encoder is not mentioned then raise an error message.
#     for feat_info in features_info:
#         if feat_info['encoder']:
#             feat_needing_encoder = feat_info['name']
#             flag = 0
#             for model_info in models_info:
#                 encoders = model_info.get(f'{entity}_encoder', [])
#                 for encoder in encoders:
#                     if encoder['feat'] == feat_needing_encoder:
#                         flag = 1
#                 assert flag == 1, print(f'encoder information missing for feature:'
#                                         f'{feat_needing_encoder}')

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


def create_file_prefix(params, select_dfeat_dict, select_cfeat_dict, split_type):
    dir_prefix = f"{params.out_dir}/k_{params.abundance}/{split_type}//"
    dfeat_names = select_dfeat_dict['mtx'].keys()
    cfeat_names = select_cfeat_dict['mtx'].keys()

    dfeat_str = 'D_'
    cfeat_str = 'C_'
    for feat_name in dfeat_names:
        encoder_str = select_dfeat_dict['encoder'].get(feat_name,'')
        encoder_str = f"_{encoder_str}" if encoder_str else encoder_str

        preprocess_str = select_dfeat_dict['preprocess'].get(feat_name,'')
        preprocess_str = f"_{preprocess_str}" if preprocess_str else preprocess_str

        norm_str = select_dfeat_dict['norm'].get(feat_name,'')
        norm_str=f"_{norm_str}" if norm_str else norm_str
        dfeat_str = dfeat_str + f'{feat_name}{encoder_str}{preprocess_str}{norm_str}_'

    for feat_name in cfeat_names:
        encoder_str = select_cfeat_dict['encoder'].get(feat_name,'')
        encoder_str = f"_{encoder_str}" if encoder_str else encoder_str
        preprocess_str = select_cfeat_dict['preprocess'].get(feat_name,'')
        preprocess_str = f"_{preprocess_str}" if preprocess_str else preprocess_str
        norm_str = select_cfeat_dict['norm'].get(feat_name,'')
        norm_str=f"_{norm_str}" if norm_str else norm_str

        cfeat_str = cfeat_str + f'{feat_name}{encoder_str}{preprocess_str}{norm_str}_'


    feat_model_prefix = (dfeat_str + cfeat_str).strip('_')
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
    hp.update(decoder['hp'])

    drug_encoders = model_info.get('drug_encoder',[])
    cell_encoders = model_info.get('cell_encoder',[])

    for drug_encoder in drug_encoders if drug_encoders is not None else [] :
        hp.update(drug_encoder['hp'])

    for cell_encoder in cell_encoders if cell_encoders is not None else [] :
        hp.update(cell_encoder['hp'])

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

def mol_graph_to_GCN_data(mol_graph_dict):
    '''convert atom features and adjacency list of each drug molecule into a data compatible with
    training pytorch geometric models'''
    mol_gcn_data_dict={}
    for pid in mol_graph_dict:
        mol_feat = mol_graph_dict[pid][0]
        mol_feat_dim = mol_feat.shape[1]
        c_size = mol_feat.shape[0]
        adj_list = mol_graph_dict[pid][1]
        edges = adjacency_list_to_edges(adj_list)
        GCNData = DATA.Data(x=torch.Tensor(mol_feat),
                    edge_index=torch.LongTensor(edges).transpose(1, 0)
                    if len(edges)>0 else torch.empty((2, 0), dtype=torch.long))
        GCNData.__setitem__('c_size', torch.LongTensor([c_size]))
        mol_gcn_data_dict[str(pid)] = GCNData
    return mol_gcn_data_dict, mol_feat_dim



def adjacency_list_to_edges(adj_list):
    edges = []
    for node, neighbors in enumerate(adj_list):
        for neighbor in neighbors:
            edges.append([node, neighbor])
    return edges

