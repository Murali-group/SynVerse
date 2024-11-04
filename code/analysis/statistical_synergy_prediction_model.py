import pandas as pd
import numpy as np
def get_mean_node_edge_type_scores(edges):
    node_edge_type_scores = {}
    for node1, node2, edge_type, score in edges:
        if (node1, edge_type) not in node_edge_type_scores:
            node_edge_type_scores[(node1, edge_type)] = []
        if (node2, edge_type) not in node_edge_type_scores:
            node_edge_type_scores[(node2, edge_type)] = []
        node_edge_type_scores[(node1, edge_type)].append(score)
        node_edge_type_scores[(node2, edge_type)].append(score)
    #compute mean
    node_edge_type_counts = {(node, edge_type): len(node_edge_type_scores[(node, edge_type)]) for (node, edge_type) in node_edge_type_scores }
    node_edge_type_mean_scores = {(node, edge_type): np.mean(node_edge_type_scores[(node, edge_type)]) for (node, edge_type) in node_edge_type_scores }


    return node_edge_type_mean_scores, node_edge_type_counts


def statistical_model(train_df, test_df, score_name):

    '''
    Compute  node_edgetype_score_dict such that it contains the mean score for each (node, edge_type) pair present in train_df.
    The node can be a source or target node as this is an undirected network.
    Now, for each (source, target, edge_type) tuple present in test_df, the predicted score would be  the average of
    node_edgetype_score_dict.get((source,edge_type)) and node_edgetype_score_dict.get((target,edge_type))
    '''
    train_edges = list(zip(train_df['source'],train_df['target'], train_df['edge_type'], train_df[score_name]))
    node_edge_type_score_dict, node_edge_type_counts_dict = get_mean_node_edge_type_scores(train_edges)

    pres_scores = []
    for i, row in test_df.iterrows():
        s = node_edge_type_score_dict.get((row['source'], row['edge_type']),np.nan)
        t = node_edge_type_score_dict.get((row['target'], row['edge_type']),np.nan)
        score_arr = np.array([s,t])

        #taking weighted mean as pred score
        s_count = node_edge_type_counts_dict.get((row['source'], row['edge_type']), 0)
        t_count = node_edge_type_counts_dict.get((row['target'], row['edge_type']), 0)

        norm_weights = np.array([s_count, t_count])/np.array([s_count, t_count]).sum()
        mask = ~np.isnan(score_arr)
        pred_score = np.average(score_arr[mask], weights = norm_weights[mask])
        pres_scores.append(pred_score)

        #taking mean as predicted score
        # pred_score = np.nanmean(score_arr)
        # pres_scores.append(pred_score)

    test_df[f'pred_{score_name}'] = pres_scores
    test_df[f'pred_{score_name}'].fillna(0, inplace=True)

    #now compute the mean square error between true and predicted score.
    mse = ((test_df[score_name]-test_df[f'pred_{score_name}'])**2).mean()
    print(f"Mean Squared Error on Test data: {mse}")


