import numpy as np
import networkx as nx
import scipy.sparse as sp
import random
import models.gnn.gnn_utils as utils
import copy
import pickle
import os

def create_cross_val_split_non_drug_drug_edges(number_of_folds, adj_mats_init, non_drug_drug_edge_types, total_cell_lines,cross_val_dir):
    folds = {edge_type: {} for edge_type in ['gene_gene','target_drug','drug_target']}
    pickle_file = cross_val_dir + '/pos_non_drug_drug.pkl'

    if not os.path.exists(pickle_file):
        for edge_type in non_drug_drug_edge_types:
            if edge_type == 'gene_gene':
                folds_gene_gene_edges = {cell_line_idx: {} for cell_line_idx in range(len(adj_mats_init[edge_type]))}
                for k in range(len(adj_mats_init[edge_type])):
                    edges_all, _, _ = utils.sparse_to_tuple(adj_mats_init[edge_type][k])
                    edges_set = set(map(tuple, edges_all))

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
                        folds_gene_gene_edges[k][i] = val_edges
                folds['gene_gene'] = folds_gene_gene_edges

            elif (edge_type == 'drug_target'):
                folds_drug_target_edges = {idx: {} for idx in range(len(adj_mats_init[edge_type]))}
                folds_target_drug_edges = {idx: {} for idx in range(len(adj_mats_init[edge_type]))}

                for k in range(len(adj_mats_init[edge_type])):
                    edges_all, _, _ = utils.sparse_to_tuple(adj_mats_init[edge_type][k])
                    edges_set = set(map(tuple, edges_all))

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
                        folds_drug_target_edges[k][i] = val_edges_1_2
                        folds_target_drug_edges[k][i] = val_edges_2_1
                folds['drug_target'] = folds_drug_target_edges
                folds['target_drug'] = folds_target_drug_edges
        with open(pickle_file, 'wb') as handle:
            pickle.dump(folds, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(pickle_file, 'rb') as handle:
        folds = pickle.load(handle)

    # for edge_type in cross_val_folds_non_drug_drug_edges:
    #     for fold_no in cross_val_folds_non_drug_drug_edges[edge_type]:
    #         print('pos_edges: ', edge_type, ': ', len(cross_val_folds_non_drug_drug_edges[edge_type][fold_no]))

    return folds

def create_degree_based_neg_cross_val_split_non_drug_drug_edges(number_of_folds, adj_mats_orig, non_drug_drug_edge_types, neg_fact, cross_val_dir):

    pickle_file = cross_val_dir + '/neg_non_drug_drug.pkl'

    if not os.path.exists(pickle_file):
        neg_cross_val_folds_non_drug_drug_edges = {edge_type: {} for edge_type in ['gene_gene','target_drug','drug_target']}
        for edge_type in non_drug_drug_edge_types:
            specific_adj_mat = adj_mats_orig[edge_type][0]

            edges_all, _, _ = utils.sparse_to_tuple(specific_adj_mat)
            edges_set = set(map(tuple, edges_all))

            # this edges set will contain only(x,y), not also(y,x)
            n_source_node = specific_adj_mat.shape[0]
            n_target_node = specific_adj_mat.shape[1]
            print('shape of gene gene adj matrix: ', n_source_node, n_target_node)

            edges_set, source_node_degrees, target_node_degrees = find_node_degrees(n_source_node, n_target_node, edges_set, edge_type)
            total_edges = len(edges_set)
            edges_per_fold = neg_fact*int(total_edges / number_of_folds + 1)
            val_edges_false = set()

            if (edge_type == 'gene_gene'):
                # sorted_edges_set = set([(max(idx_1, idx_2), min(idx_1, idx_2)) for idx_1, idx_2 in edges_set])
                print('total nodes: ', len(source_node_degrees.keys()))
                for node in source_node_degrees:
                    source_node_degree = source_node_degrees[node]
                    if source_node_degree>0:
                        total_neg_sample = source_node_degree * neg_fact
                        print('sample required for node: ', node, total_neg_sample)
                        n = neg_fact
                        while True:
                            # generate pairs having first node=node and some random node as second node. generate n times pairs than that is needed.
                            idx_i = [node] * total_neg_sample * n
                            idx_j = random.choices(list(target_node_degrees.keys()), k=total_neg_sample * n, \
                                                   weights=list(target_node_degrees.values()))
                            new_val_edges = set(zip(idx_i, idx_j))

                            # sort and #remove (x,x) pairs i.e. where two drugs are the same
                            new_val_edges = set([(max(idx_1, idx_2), min(idx_1, idx_2)) for idx_1, idx_2 in new_val_edges if
                                                 idx_1 != idx_2])
                            new_false_val_edges = new_val_edges.difference(edges_set)
                            new_false_val_edges = new_false_val_edges.difference(val_edges_false)
                            n += 1
                            print('l of new_false_val: ', len(new_false_val_edges))
                            if (len(new_false_val_edges) >= total_neg_sample):
                                val_edges_false = val_edges_false.union(set(list(new_false_val_edges)[0:total_neg_sample]))
                                # print(node)
                                break
                val_edges_false_1_2 = list(val_edges_false)
                #now split into 5 folds
                start = 0
                for ith_fold in range(number_of_folds):
                    end = start + int(edges_per_fold)
                    neg_cross_val_folds_non_drug_drug_edges[edge_type][ith_fold] = val_edges_false_1_2[start:end]
                    start = end

                if end < len(val_edges_false_1_2):
                    fold_no = random.randint(0,number_of_folds-1)
                    neg_cross_val_folds_non_drug_drug_edges[edge_type][fold_no] += \
                        val_edges_false_1_2[end:len(val_edges_false_1_2)]

                #now add the (y,x) pair with every (x,y) pairs present in each fold
                for ith_fold in range(number_of_folds):
                    l1 = neg_cross_val_folds_non_drug_drug_edges[edge_type][ith_fold]
                    neg_cross_val_folds_non_drug_drug_edges[edge_type][ith_fold] = l1 + [(idx_2, idx_1) for idx_1, idx_2 in l1]


            elif (edge_type == 'target_drug') | (edge_type == 'drug_target'):
                # edges_per_fold = neg_fact*int(total_edges / number_of_folds + 1)
                print('edge type: ', edge_type)
                print('total nodes: ', len(source_node_degrees.keys()))
                for node in source_node_degrees:
                    source_node_degree = source_node_degrees[node]
                    if source_node_degree > 0:
                        total_neg_sample = source_node_degree * neg_fact
                        print('sample required for node: ', node, total_neg_sample)
                        n = neg_fact
                        while True:
                            # generate pairs having first node=node and some random node as second node. generate n times pairs than that is needed.
                            idx_i = [node] * total_neg_sample * n
                            idx_j = random.choices(list(target_node_degrees.keys()), k=total_neg_sample * n,\
                                                   weights=list(target_node_degrees.values()))

                            new_val_edges = set(zip(idx_i, idx_j))

                            new_false_val_edges = new_val_edges.difference(edges_set)
                            new_false_val_edges = new_false_val_edges.difference(val_edges_false)
                            n += 1
                            if (len(new_false_val_edges) >= total_neg_sample):
                                val_edges_false = val_edges_false.union(set(list(new_false_val_edges)[0:total_neg_sample]))
                                # print(node)
                                break
                val_edges_false_x_y = list(val_edges_false)
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
                    neg_cross_val_folds_non_drug_drug_edges[edge_type][fold_no] +=\
                        val_edges_false_x_y[end:len(val_edges_false_x_y)]

                    if edge_type=='target_drug':
                        neg_cross_val_folds_non_drug_drug_edges['drug_target'][fold_no] += \
                            val_edges_false_y_x[end:len(val_edges_false_y_x)]
                    else:
                        neg_cross_val_folds_non_drug_drug_edges['target_drug'][fold_no] += \
                            val_edges_false_y_x[end:len(val_edges_false_y_x)]

        with open(pickle_file, 'wb') as handle:
            pickle.dump(neg_cross_val_folds_non_drug_drug_edges, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(pickle_file, 'rb') as handle:
        neg_cross_val_folds_non_drug_drug_edges = pickle.load(handle)

    for edge_type in neg_cross_val_folds_non_drug_drug_edges:
        for fold_no in neg_cross_val_folds_non_drug_drug_edges[edge_type]:
            print('neg_edges: ', edge_type , ': ',len(neg_cross_val_folds_non_drug_drug_edges[edge_type][fold_no]))

    print('neg_cross_val_folds_non_drug_drug_edges done\n')
    return neg_cross_val_folds_non_drug_drug_edges



def create_semi_random_neg_cross_val_split_non_drug_drug_edges(number_of_folds, adj_mats_orig, \
                                non_drug_drug_edge_types, neg_fact,total_cell_lines, cross_val_dir):

    pickle_file = cross_val_dir + '/neg_non_drug_drug.pkl'

    if not os.path.exists(pickle_file):
        neg_folds = {edge_type: {} for edge_type in ['gene_gene','target_drug','drug_target']}
        for edge_type in non_drug_drug_edge_types:
            if (edge_type == 'gene_gene'):
                neg_folds_gene_gene_edges = {idx: {} for idx in range(len(adj_mats_orig[edge_type]))}
                for k in range(len(adj_mats_orig[edge_type])):
                    specific_adj_mat = adj_mats_orig[edge_type][k]
                    edges_all, _, _ = utils.sparse_to_tuple(specific_adj_mat)
                    edges_set = set(map(tuple, edges_all))

                    # this edges set will contain only(x,y), not also(y,x)
                    n_source_node = specific_adj_mat.shape[0]
                    n_target_node = specific_adj_mat.shape[1]
                    print('shape of gene gene adj matrix: ', n_source_node, n_target_node)

                    edges_set, source_node_degrees, target_node_degrees = find_node_degrees(n_source_node,
                                                                                            n_target_node, edges_set,
                                                                                            edge_type)
                    total_edges = len(edges_set)
                    edges_per_fold = neg_fact * int(total_edges / number_of_folds + 1)
                    val_edges_false = set()
                    # sorted_edges_set = set([(max(idx_1, idx_2), min(idx_1, idx_2)) for idx_1, idx_2 in edges_set])
                    print('total nodes: ', len(source_node_degrees.keys()))
                    for node in source_node_degrees:
                        source_node_degree = source_node_degrees[node]
                        if source_node_degree>0:
                            total_neg_sample = source_node_degree * neg_fact
                            print('sample required for node: ', node, total_neg_sample)
                            n = neg_fact
                            while True:
                                # generate pairs having first node=node and some random node as second node. generate n times pairs than that is needed.
                                idx_i = [node] * total_neg_sample * n

                                idx_j = random.choices(range(n_target_node), k=total_neg_sample * n)
                                new_val_edges = set(zip(idx_i, idx_j))

                                # sort and #remove (x,x) pairs i.e. where two drugs are the same
                                new_val_edges = set([(max(idx_1, idx_2), min(idx_1, idx_2)) for idx_1, idx_2 in new_val_edges if
                                                     idx_1 != idx_2])
                                new_false_val_edges = new_val_edges.difference(edges_set)
                                new_false_val_edges = new_false_val_edges.difference(val_edges_false)
                                n += 1
                                print('l of new_false_val: ', len(new_false_val_edges))
                                if (len(new_false_val_edges) >= total_neg_sample):
                                    val_edges_false = val_edges_false.union(set(list(new_false_val_edges)[0:total_neg_sample]))
                                    # print(node)
                                    break
                    val_edges_false_1_2 = list(val_edges_false)
                    #now split into 5 folds
                    start = 0
                    for ith_fold in range(number_of_folds):
                        end = start + int(edges_per_fold)
                        neg_folds_gene_gene_edges[k][ith_fold] = val_edges_false_1_2[start:end]
                        start = end

                    if end < len(val_edges_false_1_2):
                        fold_no = random.randint(0,number_of_folds-1)
                        neg_folds_gene_gene_edges[k][fold_no] += \
                            val_edges_false_1_2[end:len(val_edges_false_1_2)]

                    #now add the (y,x) pair with every (x,y) pairs present in each fold
                    for ith_fold in range(number_of_folds):
                        l1 = neg_folds_gene_gene_edges[k][ith_fold]
                        neg_folds_gene_gene_edges[k][ith_fold] = l1 + [(idx_2, idx_1) for idx_1, idx_2 in l1]

                neg_folds['gene_gene'] = neg_folds_gene_gene_edges

            elif (edge_type == 'drug_target'):
                neg_folds_drug_target_edges = {idx: {} for idx in range(len(adj_mats_orig[edge_type]))}
                neg_folds_target_drug_edges = {idx: {} for idx in range(len(adj_mats_orig[edge_type]))}
                for k in range(len(adj_mats_orig[edge_type])):
                    specific_adj_mat = adj_mats_orig[edge_type][k]
                    edges_all, _, _ = utils.sparse_to_tuple(specific_adj_mat)
                    edges_set = set(map(tuple, edges_all))

                    # this edges set will contain only(x,y), not also(y,x)
                    n_source_node = specific_adj_mat.shape[0]
                    n_target_node = specific_adj_mat.shape[1]
                    print('shape of gene gene adj matrix: ', n_source_node, n_target_node)

                    edges_set, source_node_degrees, target_node_degrees = find_node_degrees(n_source_node,
                                                                                            n_target_node, edges_set,
                                                                                            edge_type)
                    total_edges = len(edges_set)
                    edges_per_fold = neg_fact * int(total_edges / number_of_folds + 1)
                    val_edges_false = set()
                    print('edge type: ', edge_type)
                    print('total nodes: ', len(source_node_degrees.keys()))


                    for node in source_node_degrees:
                        source_node_degree = source_node_degrees[node]
                        if source_node_degree > 0:
                            total_neg_sample = source_node_degree * neg_fact
                            print('sample required for node: ', node, total_neg_sample)
                            n = neg_fact
                            while True:
                                # generate pairs having first node=node and some random node as second node. generate n times pairs than that is needed.
                                idx_i = [node] * total_neg_sample * n
                                idx_j = random.choices(range(n_target_node), k=total_neg_sample * n)
                                new_val_edges = set(zip(idx_i, idx_j))

                                new_false_val_edges = new_val_edges.difference(edges_set)
                                new_false_val_edges = new_false_val_edges.difference(val_edges_false)
                                n += 1
                                if (len(new_false_val_edges) >= total_neg_sample):
                                    val_edges_false = val_edges_false.union(set(list(new_false_val_edges)[0:total_neg_sample]))
                                    # print(node)
                                    break
                    val_edges_false_x_y = list(val_edges_false)
                    # if this is (0,1) edge type, then set negative validation edge sampling for (1,0) type edge as well.
                    val_edges_false_y_x = [(idx_2, idx_1) for idx_1, idx_2 in val_edges_false_x_y]

                    # now split into 5 folds
                    start = 0
                    for ith_fold in range(number_of_folds):
                        end = start + edges_per_fold
                        neg_folds_drug_target_edges[k][ith_fold] = val_edges_false_x_y[start:end]
                        neg_folds_target_drug_edges[k][ith_fold] = \
                            val_edges_false_y_x[start:end]
                        start = end
                    if end < len(val_edges_false_x_y):
                        fold_no = random.randint(0, number_of_folds - 1)
                        neg_folds_drug_target_edges[k][fold_no] +=\
                            val_edges_false_x_y[end:len(val_edges_false_x_y)]

                        neg_folds_target_drug_edges[k][fold_no] += \
                            val_edges_false_y_x[end:len(val_edges_false_y_x)]
                neg_folds['drug_target'] = neg_folds_drug_target_edges
                neg_folds['target_drug'] = neg_folds_target_drug_edges

        with open(pickle_file, 'wb') as handle:
            pickle.dump(neg_folds, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(pickle_file, 'rb') as handle:
        neg_folds = pickle.load(handle)
    #
    # for edge_type in neg_cross_val_folds_non_drug_drug_edges:
    #     for fold_no in neg_cross_val_folds_non_drug_drug_edges[edge_type]:
    #         print('neg_edges: ', edge_type , ': ',len(neg_cross_val_folds_non_drug_drug_edges[edge_type][fold_no]))

    print('neg_cross_val_folds_non_drug_drug_edges done\n')
    return neg_folds

def find_node_degrees(n_source_node, n_target_node, edges_set, edge_type):

    source_node_appearance = {node: 0 for node in range(n_source_node)}
    node_degrees = {node: 0 for node in range(n_target_node)}
    if(edge_type=='gene_gene'):
        edges_set = set([(max(idx_1, idx_2), min(idx_1, idx_2)) for idx_1, idx_2 in edges_set])
        for idx_1, idx_2 in edges_set:
            source_node_appearance[idx_1] += 1
            node_degrees[idx_1] += 1
            node_degrees[idx_2] += 1

    elif (edge_type == 'drug_target')|(edge_type == 'target_drug'):
        for idx_1, idx_2 in edges_set:
            source_node_appearance[idx_1] += 1
            node_degrees[idx_2] += 1

    return edges_set, source_node_appearance, node_degrees
