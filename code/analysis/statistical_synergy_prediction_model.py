import copy
import pandas as pd
import numpy as np
import random

def compute_degree_dist(train_df, score_name):
    drugs = set(train_df['source']).union(set(train_df['target']))
    dist_dict = {}
    for drug in drugs:
        # Compute probability distributions using histogram (for discrete values)
        synergy_values = train_df[(train_df['source']==drug)|(train_df['target']==drug)][score_name].values
        hist_X, bin_edges_X = np.histogram(synergy_values, bins=100, density=True)

        # Compute bin centers
        X_bins = (bin_edges_X[:-1] + bin_edges_X[1:]) / 2

        # Normalize to ensure probabilities sum to 1 (if needed)
        f_X = hist_X / np.sum(hist_X)
        dist_dict[drug] = f_X
    return dist_dict

def edge_type_spec_joint_node_degree_based_model(train_df, test_df, score_name, choice = 'sample'):
    '''
    Compute  node_edgetype_score_dict such that it contains the mean score for each (node, edge_type) pair present in train_df.
    The node can be a source or target node as this is an undirected network.
    Now, for each (source, target, edge_type) tuple present in test_df, the predicted score would be  the average of
    node_edgetype_score_dict.get((source,edge_type)) and node_edgetype_score_dict.get((target,edge_type))
    '''
    pred_scores = []

    #for each drug in training compute the distribution of synergy scores associated with it.
    node_score_dist= compute_degree_dist(train_df, score_name)
    for i, row in test_df.iterrows():
        d1 = row['source']
        d2 = row['target']


        pred_score = np.outer(node_score_dist[d1], node_score_dist[d2])  # g(X, Y) = f(X) * f(Y)




    #     pred_scores.append(pred_score)
    #
    # test_df[f'pred_{score_name}'] = pred_scores
    # test_df[f'pred_{score_name}'].fillna(0, inplace=True)
    #
    # #now compute the mean square error between true and predicted score.
    # mse = ((test_df[score_name]-test_df[f'pred_{score_name}'])**2).mean()
    # print(f"Mean Squared Error on Test data: {mse}")
    # return pred_scores, mse


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
    node_edge_type_sampled_scores = {(node, edge_type): random.choice(node_edge_type_scores[(node, edge_type)]) for (node, edge_type) in node_edge_type_scores }


    return node_edge_type_mean_scores,node_edge_type_sampled_scores, node_edge_type_counts

def node_degree_based_sampling_model(train_df, test_df, score_name):
    '''
        Rule for prediction on test: For all the triplets belonging to a certain edge type,
                                    predict score by sampling score uniformly at random from training scores belonging to the same edge type.
    '''
    test_df['ID'] = range(len(test_df))
    test_subset_drug_pairs = list(zip(test_df['source'],test_df['target']))
    predicted_scores = []
    for d1, d2 in test_subset_drug_pairs:
        # sample from train
        d1_train_scores = train_df[train_df['source'] == d1][score_name].values
        d2_train_scores = train_df[train_df['target'] == d2][score_name].values
        if ((len(d1_train_scores)>0) & (len(d2_train_scores)>0)):
            predicted_score = np.mean([random.choice(d1_train_scores),
                                     random.choice(d2_train_scores)])
        elif ((len(d1_train_scores)==0) & (len(d2_train_scores)==0)):
            predicted_score = random.choice(train_df[score_name].values)
        elif (len(d1_train_scores)>0):
            predicted_score = random.choice(d1_train_scores)
        else:
            predicted_score = random.choice(d2_train_scores)

        predicted_scores.append(predicted_score)
    test_df['predicted'] = predicted_scores

    mse_loss = np.mean(((test_df[score_name]-test_df['predicted'])**2))

    return list(test_df['predicted']), mse_loss


def edge_type_spec_node_degree_based_sampling_model(train_df, test_df, score_name):
    '''
        Rule for prediction on test: For all the triplets belonging to a certain edge type,
                                    predict score by sampling score uniformly at random from training scores belonging to the same edge type.
    '''
    test_df['ID'] = range(len(test_df))
    test_edge_types = test_df['edge_type'].unique()

    predicted_df = pd.DataFrame()
    for edge_type in test_edge_types:
        test_subset_df = copy.deepcopy(test_df[test_df['edge_type'] == edge_type])
        test_subset_drug_pairs = list(zip(test_subset_df['source'],test_subset_df['target']))
        train_subset = train_df[train_df['edge_type'] == edge_type]
        predicted_scores = []
        for d1, d2 in test_subset_drug_pairs:
            # sample from train
            d1_train_scores = train_subset[train_subset['source'] == d1][score_name].values
            d2_train_scores = train_subset[train_subset['target'] == d2][score_name].values
            if ((len(d1_train_scores)>0) & (len(d2_train_scores)>0)):
                predicted_score = np.mean([random.choice(d1_train_scores),
                                         random.choice(d2_train_scores)])
            elif ((len(d1_train_scores)==0) & (len(d2_train_scores)==0)):
                predicted_score = random.choice(train_subset[score_name].values)
            elif (len(d1_train_scores)>0):
                predicted_score = random.choice(d1_train_scores)
            else:
                predicted_score = random.choice(d2_train_scores)

            predicted_scores.append(predicted_score)
        test_subset_df['predicted'] = predicted_scores
        predicted_df = pd.concat([predicted_df, test_subset_df])

    predicted_df=predicted_df.sort_values(by='ID', ascending=True)
    mse_loss = np.mean(((predicted_df[score_name]-predicted_df['predicted'])**2))

    return list(predicted_df['predicted']), mse_loss


def edge_type_spec_node_degree_based_avg_model(train_df, test_df, score_name):

    '''
    Compute  node_edgetype_score_dict such that it contains the mean score for each (node, edge_type) pair present in train_df.
    The node can be a source or target node as this is an undirected network.
    Now, for each (source, target, edge_type) tuple present in test_df, the predicted score would be  the average of
    node_edgetype_score_dict.get((source,edge_type)) and node_edgetype_score_dict.get((target,edge_type))
    '''
    train_edges = list(zip(train_df['source'],train_df['target'], train_df['edge_type'], train_df[score_name]))
    node_edge_type_score_dict, node_edge_type_counts_dict = get_mean_node_edge_type_scores(train_edges)

    pred_scores = []
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
        pred_scores.append(pred_score)

        #taking mean as predicted score
        # pred_score = np.nanmean(score_arr)
        # pres_scores.append(pred_score)

    test_df[f'pred_{score_name}'] = pred_scores
    test_df[f'pred_{score_name}'].fillna(0, inplace=True)

    #now compute the mean square error between true and predicted score.
    mse = ((test_df[score_name]-test_df[f'pred_{score_name}'])**2).mean()
    print(f"Mean Squared Error on Test data: {mse}")
    return pred_scores, mse

def edge_type_spec_node_degree_based_model(train_df, test_df, score_name, choice = 'sample'):

    '''
    Compute  node_edgetype_score_dict such that it contains the mean score for each (node, edge_type) pair present in train_df.
    The node can be a source or target node as this is an undirected network.
    Now, for each (source, target, edge_type) tuple present in test_df, the predicted score would be  the average of
    node_edgetype_score_dict.get((source,edge_type)) and node_edgetype_score_dict.get((target,edge_type))
    '''
    train_edges = list(zip(train_df['source'],train_df['target'], train_df['edge_type'], train_df[score_name]))
    node_edge_type_mean_score_dict,node_edge_type_sampled_score_dict, node_edge_type_counts_dict = get_mean_node_edge_type_scores(train_edges)

    if choice=='sample':#sample score for certain drug
        node_edge_type_score_dict = node_edge_type_sampled_score_dict
    elif choice=='average':
        node_edge_type_score_dict = node_edge_type_mean_score_dict

    pred_scores = []
    for i, row in test_df.iterrows():
        s = node_edge_type_score_dict.get((row['source'], row['edge_type']),np.nan)
        t = node_edge_type_score_dict.get((row['target'], row['edge_type']),np.nan)
        score_arr = np.array([s,t])

        #taking one of the source or target's pred score based on their frequency
        s_count = node_edge_type_counts_dict.get((row['source'], row['edge_type']), 0)
        t_count = node_edge_type_counts_dict.get((row['target'], row['edge_type']), 0)

        norm_weights = np.array([s_count, t_count])/np.array([s_count, t_count]).sum()
        pred_score = np.random.choice(score_arr, 1, p=norm_weights)
        pred_scores.append(pred_score)

    test_df[f'pred_{score_name}'] = pred_scores
    test_df[f'pred_{score_name}'].fillna(0, inplace=True)

    #now compute the mean square error between true and predicted score.
    mse = ((test_df[score_name]-test_df[f'pred_{score_name}'])**2).mean()
    print(f"Mean Squared Error on Test data: {mse}")
    return pred_scores, mse


# def label_distribution_based_model(train_df, test_df, score_name, split_type='leave_drug'):
#     if split_type=='leave_drug':
#         '''
#             Rule for prediction on test: For all the triplets belonging to a certain edge type,
#                                         predict score by sampling score uniformly at random from training scores belonging to the same edge type.
#         '''
#         test_df['ID'] = range(len(test_df))
#         test_edge_types = test_df['edge_type'].unique()
#         predicted_df = pd.DataFrame()
#         for edge_type in test_edge_types:
#             test_subset_df = test_df[test_df['edge_type'] == edge_type]
#
#             #sample from train
#             train_subset_scores = list(train_df[train_df['edge_type'] == edge_type][score_name])
#
#             # Check if there are any train scores for the edge type
#             if len(train_subset_scores) > 0:
#                 # Sample uniformly at random from the training scores
#                 test_subset_df['predicted'] = random.choices(train_subset_scores, k=len(test_subset_df))
#             else:
#                 # Handle the case where no training scores exist for the edge type
#                 train_scores = train_df[score_name]
#                 test_subset_df['predicted'] =  random.choices(list(train_scores), k=len(test_subset_df))
#
#             predicted_df = pd.concat([predicted_df, test_subset_df])
#
#         predicted_df=predicted_df.sort_values(by='ID', ascending=True)
#         mse_loss = np.mean(((predicted_df[score_name]-predicted_df['predicted'])**2))
#
#         return list(predicted_df['predicted']), mse_loss

def label_distribution_based_model(train_df, test_df, score_name):
        '''
            Rule for prediction on test: sampling score uniformly at random from training scores
        '''
        train_scores = train_df[score_name]
        test_df['predicted'] =  random.choices(list(train_scores), k=len(test_df))
        mse_loss = np.mean(((test_df[score_name]-test_df['predicted'])**2))

        return list(test_df['predicted']), mse_loss


