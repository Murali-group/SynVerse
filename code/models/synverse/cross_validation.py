import numpy as np
import networkx as nx
import scipy.sparse as sp
import random
import models.synverse.utils as utils

def create_cross_val_split_non_drug_drug_edges(number_of_folds, adj_mats_init, non_drug_drug_edge_types):

    cross_val_folds_non_drug_drug_edges = {edge_type: {} for edge_type in non_drug_drug_edge_types}

    for edge_type in non_drug_drug_edge_types:
        edges_all, _, _ = utils.sparse_to_tuple(adj_mats_init[edge_type][0])
        #only one gene-gene network, one drug-target network and one target-drug network, so [edge_type][0]=>index is going to be zero always
        edges_set = set(map(tuple, edges_all))
        if edge_type == 'gene_gene':
            # keep only (x,y) pair and get rid of (y,x) pair
            edges_directed_set = set([(max(idx_1, idx_2), min(idx_1, idx_2)) for idx_1, idx_2 in edges_set])
            num_half_val = int(len(edges_directed_set)/number_of_folds + 1)
            for i in range(number_of_folds):
                # print('gene gene cross val: ',i)
                if len(edges_directed_set) > num_half_val:
                    val_edges_1_2 = set(random.sample(edges_directed_set, num_half_val))
                else:
                    val_edges_1_2 = edges_directed_set
                edges_directed_set = edges_directed_set.difference(val_edges_1_2)

                val_edges_2_1 = set([(idx_2, idx_1) for idx_1, idx_2 in val_edges_1_2])
                val_edges = val_edges_1_2.union(val_edges_2_1)

                val_edges = list(val_edges)
                cross_val_folds_non_drug_drug_edges[edge_type][i] = val_edges

        elif (edge_type == 'target_drug')|(edge_type == 'drug_target'):

            num_val = int(len(edges_set) / number_of_folds+1)
            for i in range(number_of_folds):
                # print('drug gene cross val: ', i)
                if len(edges_set) > num_val:
                    val_edges_1_2 = set(random.sample(edges_set, num_val))
                else:
                    val_edges_1_2 = edges_set
                edges_set = edges_set.difference(val_edges_1_2)


                #if edge type == (0,1) also fill up the cross_val_splits for (1,0) type edges
                #if edge type == (1,0) also fill up the cross_val_splits for (0,1) type edges
                val_edges_2_1 = [(idx_2, idx_1) for idx_1, idx_2 in val_edges_1_2]

                val_edges_1_2 = list(val_edges_1_2)
                val_edges_2_1 = list(val_edges_2_1)
                cross_val_folds_non_drug_drug_edges[edge_type][i] = val_edges_1_2

                if(edge_type=='target_drug'):
                    cross_val_folds_non_drug_drug_edges ['drug_target'][i]= val_edges_2_1
                else:
                    cross_val_folds_non_drug_drug_edges['target_drug'][i] = val_edges_2_1

    return cross_val_folds_non_drug_drug_edges

def create_neg_cross_val_split_non_drug_drug_edges(number_of_folds, adj_mats_orig, non_drug_drug_edge_types):
    neg_cross_val_folds_non_drug_drug_edges = {edge_type: {} for edge_type in non_drug_drug_edge_types}
    for edge_type in non_drug_drug_edge_types:
        edges_all, _, _ = utils.sparse_to_tuple(adj_mats_orig[edge_type][0])
        edges_set = set(map(tuple, edges_all))
        total_edges = len(edges_set)
        edges_per_fold = int(total_edges/number_of_folds + 1)
        val_edges_false=set()
        if (edge_type == 'gene_gene'):
            while True:
                # print("Constructing val edges=", "%04d/%04d" % (len(val_edges_false), len(val_edges)))
                # sample #num_val  indexes at a time
                idx_i = np.random.randint(0, adj_mats_orig[edge_type][0].shape[0], size=int(total_edges / 2))
                idx_j = np.random.randint(0, adj_mats_orig[edge_type][0].shape[1], size=int(total_edges / 2))

                new_val_edges = set(zip(idx_i, idx_j))
                #sort so that either  (x,y) or (y,x) remains
                new_val_edges = set([(max(idx_1, idx_2), min(idx_1, idx_2)) for idx_1, idx_2 in new_val_edges])
                new_false_val_edges = new_val_edges.difference(edges_set)
                val_edges_false = val_edges_false.union(new_false_val_edges)


                if len(val_edges_false) >= int(total_edges / 2):
                    # print('Val false edges done')
                    val_edges_false_1_2 = list(val_edges_false)[0:int(total_edges / 2)]
                    # val_edges_false_2_1 = [(idx_2, idx_1) for idx_1, idx_2 in val_edges_false_1_2]
                    # val_edges_false = np.array(val_edges_false_1_2 + val_edges_false_2_1)

                    #now split into 5 folds
                    start=0
                    for ith_fold in range(number_of_folds):
                        end = start + int(edges_per_fold/2)
                        neg_cross_val_folds_non_drug_drug_edges[edge_type][ith_fold] = val_edges_false_1_2[start:end]
                        start = end
                    if end < len(val_edges_false_1_2):
                        fold_no = random.randint(0,number_of_folds-1)
                        neg_cross_val_folds_non_drug_drug_edges[edge_type][fold_no]+= val_edges_false_1_2[end:len(val_edges_false_1_2)]

                    #now add the (y,x) pair with every (x,y) pairs present in each fold
                    for ith_fold in range(number_of_folds):
                        l1 = neg_cross_val_folds_non_drug_drug_edges[edge_type][ith_fold]
                        neg_cross_val_folds_non_drug_drug_edges[edge_type][ith_fold] = l1 + [(idx_2, idx_1) for idx_1, idx_2 in  l1]
                    break


        elif (edge_type == 'target_drug') | (edge_type == 'drug_target'):
            while True:
                # print("Constructing val edges=", "%04d/%04d" % (len(val_edges_false), len(val_edges)))
                # sample #num_val  indexes at a time
                idx_i = np.random.randint(0, adj_mats_orig[edge_type][0].shape[0], size=total_edges)
                idx_j = np.random.randint(0, adj_mats_orig[edge_type][0].shape[1], size=total_edges)

                new_val_edges = set(zip(idx_i, idx_j))
                new_false_val_edges = new_val_edges.difference(edges_set)
                val_edges_false = val_edges_false.union(new_false_val_edges)


                if len(val_edges_false) >= total_edges:
                    # print('Val false edges done')
                    val_edges_false_x_y = list(val_edges_false)[0:total_edges]

                    # if this is (0,1) edge type, then set negative validation edge sampling for (1,0) type edge as well.
                    val_edges_false_y_x = [(idx_2, idx_1) for idx_1, idx_2 in val_edges_false_x_y]

                    # now split into 5 folds
                    start = 0
                    for ith_fold in range(number_of_folds):
                        end = start + edges_per_fold
                        neg_cross_val_folds_non_drug_drug_edges[edge_type][ith_fold] = val_edges_false_x_y[start:end]
                        if edge_type=='target_drug':
                            neg_cross_val_folds_non_drug_drug_edges['drug_target'][ith_fold] =\
                            val_edges_false_y_x[start:end]
                        else:
                            neg_cross_val_folds_non_drug_drug_edges['target_drug'][ith_fold] = \
                                val_edges_false_y_x[start:end]
                        start = end
                    if end < len(val_edges_false_x_y):
                        fold_no = random.randint(0, number_of_folds - 1)
                        neg_cross_val_folds_non_drug_drug_edges[edge_type][fold_no] += val_edges_false_x_y[end:len(val_edges_false_x_y)]

                        if edge_type=='target_drug':
                            neg_cross_val_folds_non_drug_drug_edges['drug_target'][fold_no] += \
                                val_edges_false_y_x[end:len(val_edges_false_y_x)]
                        else:
                            neg_cross_val_folds_non_drug_drug_edges['target_drug'][fold_no] += \
                                val_edges_false_y_x[end:len(val_edges_false_y_x)]

                    break

    return neg_cross_val_folds_non_drug_drug_edges
