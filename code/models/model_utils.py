import pandas as pd
import numpy as np

#***************************************************** FEATURE PREP ************************

def get_vocab_smiles(smiles_df):
    # Create a vocabulary of characters
    vocab = sorted(set(''.join(smiles_df['smiles'].values)))
    char_to_idx = {char: idx + 1 for idx, char in enumerate(vocab)}

    char_to_idx['[PAD]'] = 0  # Padding token

    def tokenize(smiles, char_to_idx):
        return [char_to_idx[char] for char in smiles]

    smiles_df['tokenized'] = smiles_df['smiles'].apply(lambda x: tokenize(x, char_to_idx))

    # def pad_or_truncate(seq, max_len):
    #
    #     if len(seq) < max_len:
    #         return seq + [0] * (max_len - len(seq))
    #     else:
    #         return seq[:max_len]
    #
    # smiles_df['tokenized'] = smiles_df['tokenized'].apply(lambda x: pad_or_truncate(x, max_len))

    return smiles_df, len(char_to_idx)



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
