import pandas as pd
import numpy as np
from torch_geometric import data as DATA
import torch

#********************************** SYNERGY TRIPLETS ***********************************
def filter_triplets(synergy_df, dfeat_dict, cfeat_dict, feature='must', k=-1):
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

    n_after_feat_filt =len(synergy_df)
    if k>0: #keep only top k cell lines having the most synergy triplets.
        top_cell_lines = synergy_df['cell_line_name'].value_counts().nlargest(k).index
        print('top cell lines:' , top_cell_lines)
        synergy_df = synergy_df[synergy_df['cell_line_name'].isin(top_cell_lines)]

        print(f'keeping top {k} cell lines, retrieved frac:{len(synergy_df)/n_after_feat_filt}')
    #assert that there is no duplicate triplets in synergy_df
    triplets = list(zip(synergy_df['drug_1_pid'],synergy_df['drug_2_pid'],synergy_df['cell_line_name']))
    assert len(set(triplets))==len(triplets), print('still some duplicates remaining')

    drug_pids = set(synergy_df['drug_1_pid']).union(set(synergy_df['drug_2_pid']))
    cell_line_names = set(synergy_df['cell_line_name'])
    print(f'After filtering: \n#of triplets : {len(synergy_df)},\n#drugs {len(drug_pids)},'
          f' \n#cell lines {len(cell_line_names)}')
    return synergy_df


#***************************************************** FEATURE PREP ************************

def adjacency_list_to_edges(adj_list):
    edges = []
    for node, neighbors in enumerate(adj_list):
        for neighbor in neighbors:
            edges.append([node, neighbor])
    return edges

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





def concatenate_features(feat_dict, identifier_col,numeric_idx_map):
    '''
    function:
    1. Concat drug feature dfs and create a drug feature matrix containing all features of drug. Sort the matrix
    according to drug idx.
    2. Concat all cell line feature dfs and create a cell line feature matrix containing all features of cell lines
    Sort the matrix according to cell line idx.
    :return: Return drug and cell line feature matrix
    '''
    feat = pd.DataFrame()
    for feat_name in feat_dict:
        df = feat_dict[feat_name]
        # TODO remove the line to drop duplicates later
        df = df.drop_duplicates()
        feat = pd.concat([feat, df.set_index(identifier_col)], axis=1)
    feat.reset_index(inplace=True)
    feat['idx'] = feat[identifier_col].astype(str).apply(lambda x: numeric_idx_map.get(x))
    feat.dropna(inplace=True)
    feat['idx'] = feat['idx'].astype(int)
    feat.set_index('idx', inplace=True)
    feat.sort_index(inplace=True)  # sort according to index

    # assert that all index from 0 to len(dfeat) is present in order.
    assert list(feat.index) == list(range(len(feat))), print('index not in order.')
    feat.drop(identifier_col, axis=1, inplace=True)

    return feat.values

def prepare_feat_for_MLP(src_tgt_edge_df, node_X, edge_X):
    '''
    :param src_tgt_edge_df: a dataframe containing 3 columns ['source','target', 'edge_type']. Where the values are
    source_idx, target_idx, edge_type_idx.
    :param node_X: a np.array where row i contain the feature of node i, e.g.,
                    row 19 contains feature of the drug whose index is 19.
    :param edge_X: a np.array where row i contain the feature of edge_type i, e.g.,
                    row 19 contains feature of the cell line whose index is 19.
    :return: A np.array where row i corresponds to the triplets in the ith row of  src_tgt_edge_df and contains the
     concatenated features of the drugs and cell line in that triplet.
    '''
    # Extract indices for source, target, and edge type
    source_indices = src_tgt_edge_df['source'].values
    target_indices = src_tgt_edge_df['target'].values
    edge_type_indices = src_tgt_edge_df['edge_type'].values

    # Use indexing to fetch all necessary features directly
    source_features = node_X[source_indices]
    target_features = node_X[target_indices]
    edge_features = edge_X[edge_type_indices]

    # Concatenate the features along the second axis (column-wise concatenation)
    #concatenation: source-target-edge_type
    # concatenated_features = np.concatenate([source_features, target_features, edge_features], axis=1)

    concatenated_features_1 = np.concatenate([source_features, target_features, edge_features], axis=1)
    #as the edges are undirected also do the concatenation: target-source-edge_type
    concatenated_features_2 = np.concatenate([target_features,source_features,edge_features], axis=1)
    concatenated_features = np.concatenate([concatenated_features_1, concatenated_features_2], axis=0)

    return concatenated_features
