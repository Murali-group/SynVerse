import numpy as np
def normalization_wrapper(dfeat_mtx_dict, cfeat_mtx_dict, dfeat_norm_dict, cfeat_norm_dict, train_df):
    train_drug_idx = list(set(train_df['source']).union(set(train_df['target'])))
    train_cell_idx = list(set(train_df['edge_type']).union(set(train_df['edge_type'])))

    for feat_name in dfeat_norm_dict: #if feature name is present in norm dict.
        dfeat_mtx_train = dfeat_mtx_dict[feat_name][train_drug_idx, :] #keep the training data only
        if dfeat_norm_dict[feat_name] == 'std': #if norm type='std'
            mean = np.mean(dfeat_mtx_train, axis=0)
            std = np.std(dfeat_mtx_train, axis=0)
            ##normalize both train and test data with mean and std computed from train data only
            dfeat_mtx_dict[feat_name] = (dfeat_mtx_dict[feat_name]-mean)/std

    for feat_name in cfeat_norm_dict:  # if feature name is present in norm dict.
        cfeat_mtx_train = cfeat_mtx_dict[feat_name][train_cell_idx, :]  # keep the training data only
        if cfeat_norm_dict[feat_name] == 'std':  # if norm type='std'
            mean = np.mean(cfeat_mtx_train, axis=0)
            std = np.std(cfeat_mtx_train, axis=0)
            ##normalize both train and test data with mean and std computed from train data only
            cfeat_mtx_dict[feat_name] = (cfeat_mtx_dict[feat_name] - mean) / std
    return dfeat_mtx_dict, cfeat_mtx_dict


def adjacency_list_to_edges(adj_list):
    edges = []
    for node, neighbors in enumerate(adj_list):
        for neighbor in neighbors:
            edges.append([node, neighbor])
    return edges