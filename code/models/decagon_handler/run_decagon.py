from __future__ import division
from __future__ import print_function
from operator import itemgetter
from itertools import combinations
import time
import os

# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
import tensorflow as tf

import numpy as np
import networkx as nx
import scipy.sparse as sp
from scipy.io import savemat, loadmat
from sklearn import metrics
import pandas as pd
import sys
import random
# sys.path.insert(1, '/home/tasnina/Projects/Synverse')
print(sys.path)
from models.decagon_handler.decagon_minibatch import EdgeMinibatchIterator
from submodules.decagon.decagon.deep.optimizer import DecagonOptimizer
# from submodules.decagon.decagon.deep.model import DecagonModel
from models.decagon_handler.modified_decagon_model import DecagonModel
from submodules.decagon.decagon.utility import rank_metrics, preprocessing


# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
print(tf.test.is_gpu_available())

os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

np.random.seed(0)


###########################################################
#
# Functions
#
###########################################################

def sigmoid(x):
    return 1. / (1 + np.exp(-x))


def get_model_prediction(edge_type, feed_dict, placeholders, minibatch, sess, opt):
    feed_dict.update({placeholders['dropout']: 0})
    feed_dict.update({placeholders['batch_edge_type_idx']: minibatch.edge_type2idx[edge_type]})
    feed_dict.update({placeholders['batch_row_edge_type']: edge_type[0]})
    feed_dict.update({placeholders['batch_col_edge_type']: edge_type[1]})
    rec = sess.run(opt.predictions, feed_dict=feed_dict)
    return rec


def compute_drug_drug_link_probability(cell_line_specific_edges_pos, cell_line_specific_edges_neg, cell_line, rec, val_fold, idx_2_drug_node):
    #if we pass edge_type(1,1,x) then it will give sigmoid  score for drug-drug-links

        pos_edge_dict = {'drug_1_idx': [], 'drug_2_idx': [], 'cell_line':[], 'model_score': [], 'predicted': [], 'true':[], 'val_fold': []}
        neg_edge_dict = {'drug_1_idx': [], 'drug_2_idx': [], 'cell_line':[], 'model_score': [], 'predicted': [], 'true':[], 'val_fold': []}
        for u, v in cell_line_specific_edges_pos:
            pos_edge_dict['drug_1_idx'].append(u)
            pos_edge_dict['drug_2_idx'].append(v)
            pos_edge_dict['cell_line'].append(cell_line)
            pos_edge_dict['model_score'].append(rec[u,v])
            pos_edge_dict['predicted'].append(sigmoid(rec[u,v]))
            pos_edge_dict['true'].append(1)
            pos_edge_dict['val_fold'].append(val_fold)

        for u, v in cell_line_specific_edges_neg:
            neg_edge_dict['drug_1_idx'].append(u)
            neg_edge_dict['drug_2_idx'].append(v)
            neg_edge_dict['cell_line'].append(cell_line)
            neg_edge_dict['model_score'].append(rec[u, v])
            neg_edge_dict['predicted'].append(sigmoid(rec[u, v]))
            neg_edge_dict['true'].append(0)
            neg_edge_dict['val_fold'].append(val_fold)

        pos_df = pd.DataFrame.from_dict(pos_edge_dict)
        neg_df = pd.DataFrame.from_dict(neg_edge_dict)

        pos_df['drug_1'] = pos_df['drug_1_idx'].apply(lambda x: idx_2_drug_node[x])
        pos_df['drug_2'] = pos_df['drug_2_idx'].apply(lambda x: idx_2_drug_node[x])

        neg_df['drug_1'] = neg_df['drug_1_idx'].apply(lambda x: idx_2_drug_node[x])
        neg_df['drug_2'] = neg_df['drug_2_idx'].apply(lambda x: idx_2_drug_node[x])

        pos_df = pos_df[['drug_1','drug_2','cell_line', 'model_score','predicted','true', 'val_fold']].sort_values(by=['predicted'],\
                                                                                                 ascending=False)
        neg_df = neg_df[['drug_1', 'drug_2','cell_line', 'model_score', 'predicted', 'true', 'val_fold']].sort_values(by=['predicted'],\
                                                                                                    ascending=False)
        return pos_df, neg_df


# def save_drug_drug_link_probability(pos_df, neg_df, iter, out_dir):
def save_drug_drug_link_probability(pos_df, neg_df, run_, use_drug_feat_option, FLAGS, out_dir):
    lr = FLAGS.learning_rate
    epochs = FLAGS.epochs
    batch_size = FLAGS.batch_size
    dr = FLAGS.dropout
    #inputs: df with link prediction probability after applying sigmoid on models predicted score for an edges(positive, negative)
    pos_out_file = out_dir + 'run_' + str(run_) + '/'+\
                    '/pos_val_scores'+'_drugfeat_'+str(use_drug_feat_option)+'_e_'+str(epochs) +'_lr_'+str(lr) +'_batch_'+ str(batch_size) +'_dr_'+\
                   str(dr)+'.tsv'
    neg_out_file = out_dir + 'run_' + str(run_) + '/'+\
                    '/neg_val_scores'+'_drugfeat_'+str(use_drug_feat_option)+'_e_'+str(epochs) +'_lr_'+str(lr) +'_batch_'+ str(batch_size) +'_dr_'+\
                   str(dr)+'.tsv'

    os.makedirs(os.path.dirname(pos_out_file), exist_ok=True)
    os.makedirs(os.path.dirname(neg_out_file), exist_ok=True)

    pos_df.to_csv(pos_out_file, sep='\t')
    neg_df.to_csv(neg_out_file, sep='\t')

def preprocess_output_for_performance_evaluation(pos_df_all, neg_df_all):
    return pd.concat([pos_df_all,neg_df_all],axis=0)[['drug_1','drug_2','cell_line', 'model_score','predicted','true']]

def get_accuracy_scores(edges_pos, edges_neg, edge_type, adj_mats_orig, rec):

    # rec = get_model_prediction(edge_type, feed_dict, placeholders, minibatch, sess, opt)
    # print('edge type in get_accuracy_scores:', edge_type)
    # Predict on test set of edges
    preds = []
    actual = []
    predicted = []
    edge_ind = 0
    for u, v in edges_pos[edge_type[:2]][edge_type[2]]:
        score = sigmoid(rec[u, v])
        preds.append(score)
        # print(u,v)
        assert adj_mats_orig[edge_type[:2]][edge_type[2]][u, v] == 1, 'Problem 1'

        actual.append(edge_ind) #contain the index of the positive edges
        predicted.append((score, edge_ind))
        edge_ind += 1

    preds_neg = []
    for u, v in edges_neg[edge_type[:2]][edge_type[2]]:
        score = sigmoid(rec[u, v])
        preds_neg.append(score)
        if (adj_mats_orig[edge_type[:2]][edge_type[2]][u, v] != 0):
            print(edge_type, u, v)
        assert adj_mats_orig[edge_type[:2]][edge_type[2]][u, v] == 0, 'Problem 0'

        predicted.append((score, edge_ind))
        edge_ind += 1

    preds_all = np.hstack([preds, preds_neg])
    preds_all = np.nan_to_num(preds_all) #contains predicted scores of the edges according to their index
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))]) #contain true labels of edges according to their index
    predicted = list(zip(*sorted(predicted, reverse=True, key=itemgetter(0))))[1]#contain the index of the top scoring edges

    roc_sc = metrics.roc_auc_score(labels_all, preds_all)
    aupr_sc = metrics.average_precision_score(labels_all, preds_all)
    apk_sc = rank_metrics.apk(actual, predicted, k=50)

    return roc_sc, aupr_sc, apk_sc


def construct_placeholders(edge_types):
    placeholders = {
        'batch': tf.placeholder(tf.int32, name='batch'),
        'neg_batch': tf.placeholder(tf.int32, name='neg_batch'),
        'batch_edge_type_idx': tf.placeholder(tf.int32, shape=(), name='batch_edge_type_idx'),
        'batch_row_edge_type': tf.placeholder(tf.int32, shape=(), name='batch_row_edge_type'),
        'batch_col_edge_type': tf.placeholder(tf.int32, shape=(), name='batch_col_edge_type'),
        'degrees': tf.placeholder(tf.int32),
        'dropout': tf.placeholder_with_default(0., shape=()),
    }
    placeholders.update({
        'adj_mats_%d,%d,%d' % (i, j, k): tf.sparse_placeholder(tf.float32)
        for i, j in edge_types for k in range(edge_types[i, j])})
    placeholders.update({
        'feat_%d' % i: tf.sparse_placeholder(tf.float32)
        for i, _ in edge_types})
    return placeholders


def initial_model_setting(config_map):
    flags = tf.app.flags
    FLAGS = flags.FLAGS
    decagon_settings = config_map['ml_models_settings']['algs']['decagon']
    flags.DEFINE_integer('neg_sample_size', decagon_settings['neg_sample_size'], 'Negative sample size.')
    flags.DEFINE_float('learning_rate', decagon_settings['learning_rate'], 'Initial learning rate.')
    flags.DEFINE_integer('epochs', decagon_settings['epochs'], 'Number of epochs to train.')
    flags.DEFINE_integer('hidden1', 64, 'Number of units in hidden layer 1.')
    flags.DEFINE_integer('hidden2', 32, 'Number of units in hidden layer 2.')
    flags.DEFINE_float('weight_decay', decagon_settings['weight_decay'], 'Weight for L2 loss on embedding matrix.')
    flags.DEFINE_float('dropout', decagon_settings['dropout'], 'Dropout rate (1 - keep probability).')
    flags.DEFINE_float('max_margin', decagon_settings['max_margin'], 'Max margin parameter in hinge loss')
    flags.DEFINE_integer('batch_size', decagon_settings['batch_size'], 'minibatch size.')
    flags.DEFINE_boolean('bias', decagon_settings['bias'], 'Bias term.')
    return FLAGS

def check_if_a_sparse_matrix_undirected(sp_matrix):

    #check if ppi matrix contains both (x,y) and (y,x) tuple
    sp_matrix_coo = sp_matrix.tocoo()
    row = sp_matrix_coo.row
    col = sp_matrix_coo.col
    x1 = set(tuple(zip(row,col)))
    x2 = set(tuple(zip(col,row)))
    if len(x1.union(x2)) == len(x1):
        # print('all (x,y)  and (y,x) tuple present')
        return 'undirected'
    else:
        # print('Not present')
        return 'directed'
def create_cross_val_split_non_drug_drug_edges(number_of_folds, adj_mats_orig,non_drug_drug_edge_types):

    cross_val_folds_non_drug_drug_edges = {i: {} for i in range(number_of_folds)}

    for edge_type in non_drug_drug_edge_types:
        edges_all, _, _ = preprocessing.sparse_to_tuple(adj_mats_orig[edge_type][0])
        #only one gene-gene network, one drug-target network and one target-drug network, so [edge_type][0]=>index is going to be zero always
        edges_set = set(map(tuple, edges_all))
        if edge_type == (0, 0):
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
                cross_val_folds_non_drug_drug_edges[i][edge_type] = val_edges

        elif (edge_type == (0, 1))|(edge_type == (1, 0)):

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
                cross_val_folds_non_drug_drug_edges[i][edge_type] = val_edges_1_2
                cross_val_folds_non_drug_drug_edges[i] [(edge_type[1],edge_type[0])]= val_edges_2_1

    return cross_val_folds_non_drug_drug_edges

def create_neg_cross_val_split_non_drug_drug_edges(number_of_folds, adj_mats_orig, non_drug_drug_edge_types):
    neg_cross_val_folds_non_drug_drug_edges = {i: {} for i in range(number_of_folds)}
    for edge_type in non_drug_drug_edge_types:
        edges_all, _, _ = preprocessing.sparse_to_tuple(adj_mats_orig[edge_type][0])
        edges_set = set(map(tuple, edges_all))
        total_edges = len(edges_set)
        edges_per_fold = int(total_edges/number_of_folds + 1)
        val_edges_false=set()
        if (edge_type == (0,0)):
            while True:
                # print("Constructing val edges=", "%04d/%04d" % (len(val_edges_false), len(val_edges)))
                # sample #num_val  indexes at a time
                idx_i = np.random.randint(0, adj_mats_orig[edge_type][0].shape[0], size=int(total_edges / 2))
                idx_j = np.random.randint(0, adj_mats_orig[edge_type][0].shape[1], size=int(total_edges / 2))

                new_val_edges = set(zip(idx_i, idx_j))
                #sort so that either  (x,y) or (y,x) remains
                new_val_edges = set([(max(idx_1, idx_2), min(idx_1, idx_2)) for idx_1, idx_2 in \
                                      new_val_edges])
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
                        neg_cross_val_folds_non_drug_drug_edges[ith_fold][edge_type] = val_edges_false_1_2[start:end]
                        start = end
                    if end < len(val_edges_false_1_2):
                        fold_no = random.randint(0,number_of_folds-1)
                        neg_cross_val_folds_non_drug_drug_edges[fold_no][edge_type]+= val_edges_false_1_2[end:len(val_edges_false_1_2)]

                    #now add the (y,x) pair with every (x,y) pairs present in each fold
                    for ith_fold in range(number_of_folds):
                        l1 = neg_cross_val_folds_non_drug_drug_edges[ith_fold][edge_type]
                        neg_cross_val_folds_non_drug_drug_edges[ith_fold][edge_type] = l1 + [(idx_2, idx_1) for idx_1, idx_2 in  l1]
                    break


        elif (edge_type == (0, 1)) | (edge_type == (1, 0)):
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
                        neg_cross_val_folds_non_drug_drug_edges[ith_fold][edge_type] = val_edges_false_x_y[start:end]
                        neg_cross_val_folds_non_drug_drug_edges[ith_fold][(edge_type[1],edge_type[0])] =\
                            val_edges_false_y_x[start:end]
                        start = end
                    if end < len(val_edges_false_x_y):
                        fold_no = random.randint(0, number_of_folds - 1)
                        neg_cross_val_folds_non_drug_drug_edges[fold_no][edge_type] += val_edges_false_x_y[end:len(val_edges_false_x_y)]

                        neg_cross_val_folds_non_drug_drug_edges[fold_no][(edge_type[1], edge_type[0])] +=\
                            val_edges_false_y_x[end:len(val_edges_false_y_x)]
                    break

    return neg_cross_val_folds_non_drug_drug_edges



def run_decagon_model(ppi_sparse_matrix, gene_node_2_idx, drug_target_df, drug_maccs_keys_feature_df, synergy_df, non_synergy_df,\
                      cross_validation_folds_pos_drug_drug_edges, cross_validation_folds_neg_drug_drug_edges,\
                      run_, out_dir, config_map):


    number_of_folds = config_map['ml_models_settings']['cross_val']['folds']
    decagon_settings = config_map['ml_models_settings']['algs']['decagon']
    gene_adj = nx.adjacency_matrix(nx.convert_matrix.from_scipy_sparse_matrix(ppi_sparse_matrix, create_using=nx.Graph(),edge_attribute=None))
    gene_degrees = np.array(gene_adj.sum(axis=0)).squeeze()
    genes_in_ppi = gene_node_2_idx.keys()

    synergistics_drugs = set(list(synergy_df['Drug1_pubchem_cid'])).union(set(list(synergy_df['Drug2_pubchem_cid'])))
    # print('number of drugs after applying threshold on synergy data:', len(synergistics_drugs))
    drug_nodes = drug_target_df['pubchem_cid'].unique()
    drug_node_2_idx = {node: i for i, node in enumerate(drug_nodes)}
    idx_2_drug_node =  {i: node for i, node in enumerate(drug_nodes)}
    drug_target_df['gene_idx'] = drug_target_df['uniprot_id'].astype(str).apply(lambda x: gene_node_2_idx[x])
    drug_target_df['drug_idx'] = drug_target_df['pubchem_cid'].astype(str).apply(lambda x: drug_node_2_idx[x])

    # now create drug_target adjacency matrix where gene nodes are in same order as they are in gene_net
    n_drugs = len(drug_target_df['drug_idx'].unique())
    n_genes = len(genes_in_ppi)

    row = list(drug_target_df['drug_idx'])
    col = list(drug_target_df['gene_idx'])
    data = np.ones(len(row))
    drug_target_adj = sp.csr_matrix((data, (row, col)),shape=(n_drugs, n_genes))
    target_drug_adj = drug_target_adj.transpose(copy=True)

    # index all the cell lines
    cell_lines = synergy_df['Cell_line'].unique()
    cell_line_2_idx = {cell_line: i for i, cell_line in enumerate(cell_lines)}
    idx_2_cell_line = {i: cell_line for i, cell_line in enumerate(cell_lines)}
    # print('number of cell lines: ',len(cell_lines_2_idx.keys()))

    synergy_df['Drug1_idx'] = synergy_df['Drug1_pubchem_cid'].astype(str).apply(lambda x: drug_node_2_idx[x])
    synergy_df['Drug2_idx'] = synergy_df['Drug2_pubchem_cid'].astype(str).apply(lambda x: drug_node_2_idx[x])
    synergy_df['Cell_line_idx'] = synergy_df['Cell_line'].astype(str).apply(lambda x: cell_line_2_idx[x])

    # investigate/analyse result
    # pairs_per_cell_line_idx_df = synergy_df.groupby(by =['Cell_line_idx'], as_index=False).count()
    # pairs_per_cell_line_idx_df.to_csv('pairs_per_cell_line.tsv', sep='\t')
    # print('pairs_per_cell_line',pairs_per_cell_line_idx_df)

    non_synergy_df['Drug1_idx'] = non_synergy_df['Drug1_pubchem_cid'].astype(str).apply(lambda x: drug_node_2_idx[x])
    non_synergy_df['Drug2_idx'] = non_synergy_df['Drug2_pubchem_cid'].astype(str).apply(lambda x: drug_node_2_idx[x])
    non_synergy_df['Cell_line_idx'] = non_synergy_df['Cell_line'].astype(str).apply(lambda x: cell_line_2_idx[x])

    # create drug-drug synergy network for each cell line separately
    total_cell_lines = len(cell_line_2_idx.values())
    drug_drug_adj_list = []

    tagetted_genes = len(drug_target_df['uniprot_id'].unique())
    n_drugdrug_rel_types = total_cell_lines

    for cell_line_idx in range(total_cell_lines):
        #     print('cell_line_idx',cell_line_idx)
        df = synergy_df[synergy_df['Cell_line_idx'] == cell_line_idx][['Drug1_idx', 'Drug2_idx',
                                                                       'Cell_line_idx', 'Loewe_label']]
        edges = list(zip(df['Drug1_idx'], df['Drug2_idx']))

        mat = np.zeros((n_drugs, n_drugs))
        for d1, d2 in edges:
            mat[d1, d2] = mat[d2, d1] = 1.
        #         print(d1,d2)
        drug_drug_adj_list.append(sp.csr_matrix(mat))

    #### in  each drug_drug_adjacency_matrix in drug_degrees_list, if (x,y) is present then (y,x) is also present there.
    drug_degrees_list = [np.array(drug_adj.sum(axis=0)).squeeze() for drug_adj in drug_drug_adj_list]

    print('finished network building')

    print('In Final network:\n Genes:%d Targetted Genes:%d  Drugs:%d' % (n_genes, tagetted_genes, n_drugs))


    # data representation
    adj_mats_orig = {
        (0, 0): [gene_adj],
        (0, 1): [target_drug_adj],
        (1, 0): [drug_target_adj],
        (1, 1): drug_drug_adj_list,

    }
    degrees = {
        0: [gene_degrees],
        1: drug_degrees_list,
    }


    ###########################    CROSS VALIDATION PREPARATION    ######################################

    # cross validation folds contain only drug_pair index from synergy_df. Convert validation folds into list of (drug-idx, drug-idx) pairs.
    edges_all_cell_line = list(zip(synergy_df['Drug1_idx'], synergy_df['Drug2_idx'], synergy_df['Cell_line_idx']))
    # print('all cell line  edges index length: ', len(edges_all_cell_line))
    temp_cross_validation_folds = {}
    for fold in cross_validation_folds_pos_drug_drug_edges:
        # print('test: ', fold, len(cross_validation_folds[fold]), cross_validation_folds[fold])
        temp_cross_validation_folds[fold] = [edges_all_cell_line[x] for x in cross_validation_folds_pos_drug_drug_edges[fold]]
        # print(temp_cross_validation_folds[fold][0:10], cross_validation_folds_pos_drug_drug_edges[fold][0:10])
    cross_validation_folds_pos_drug_drug_edges = temp_cross_validation_folds

    # cross validation folds contain only drug_pair index from non_synergy_df. Convert validation folds into list of (drug-idx, drug-idx) pairs.
    temp_cross_validation_folds = {}
    neg_edges_all_cell_line = list(zip(non_synergy_df['Drug1_idx'], non_synergy_df['Drug2_idx'], non_synergy_df['Cell_line_idx']))
    for fold in cross_validation_folds_neg_drug_drug_edges:
        # print('test: ', fold, len(cross_validation_folds[fold]), cross_validation_folds[fold])
        temp_cross_validation_folds[fold] = [neg_edges_all_cell_line[x] for x in cross_validation_folds_neg_drug_drug_edges[fold]]
        # print(temp_cross_validation_folds[fold][0:10], cross_validation_folds_pos_drug_drug_edges[fold][0:10])
    cross_validation_folds_neg_drug_drug_edges = temp_cross_validation_folds

    non_drug_drug_edge_types = [(0,0),(0,1)]
    cross_validation_folds_non_drug_drug_edges = create_cross_val_split_non_drug_drug_edges(number_of_folds, adj_mats_orig,\
                                                                                            non_drug_drug_edge_types)
    neg_cross_validation_folds_non_drug_drug_edges = create_neg_cross_val_split_non_drug_drug_edges(number_of_folds,
                                                                                            adj_mats_orig, \
                                                                                            non_drug_drug_edge_types)


    ######################## NODE FEATURE MATRIX CREATION ###########################################

    # featureless (genes)
    gene_feat = sp.identity(n_genes)
    gene_nonzero_feat, gene_num_feat = gene_feat.shape
    gene_feat = preprocessing.sparse_to_tuple(gene_feat.tocoo())

    FLAGS = initial_model_setting(config_map)

    # features (drugs)
    use_drug_feat_options = decagon_settings['use_drug_feat']
    for use_drug_feat_option in use_drug_feat_options:
        if use_drug_feat_option:
            drug_maccs_keys_feature_df['drug_idx'] = drug_maccs_keys_feature_df['pubchem_cid'].\
                                            apply(lambda x: drug_node_2_idx[x])
            drug_maccs_keys_feature_df = drug_maccs_keys_feature_df.sort_values(by=['drug_idx'])
            assert len(drug_maccs_keys_feature_df)== n_drugs, 'problem in drug feat creation'
            drug_feat = drug_maccs_keys_feature_df.drop(columns=['pubchem_cid']).set_index('drug_idx').to_numpy()
            drug_num_feat = drug_feat.shape[1]
            drug_nonzero_feat = np.count_nonzero(drug_feat)

            drug_feat = sp.csr_matrix(drug_feat)
            drug_feat = preprocessing.sparse_to_tuple(drug_feat.tocoo())

        else:
        # #one hot encoding for drug features
            drug_feat = sp.identity(n_drugs)
            drug_nonzero_feat, drug_num_feat = drug_feat.shape
            drug_feat = preprocessing.sparse_to_tuple(drug_feat.tocoo())

        # data representation
        num_feat = {
            0: gene_num_feat,
            1: drug_num_feat,
        }
        nonzero_feat = {
            0: gene_nonzero_feat,
            1: drug_nonzero_feat,
        }
        feat = {
            0: gene_feat,
            1: drug_feat,
        }

        edge_type2dim = {k: [adj.shape for adj in adjs] for k, adjs in adj_mats_orig.items()}
        edge_type2decoder = {
            (0, 0): 'bilinear',
            (0, 1): 'bilinear',
            (1, 0): 'bilinear',
            (1, 1): 'dedicom',
        }

        edge_types = {k: len(v) for k, v in adj_mats_orig.items()}
        num_edge_types = sum(edge_types.values())
        # print("Edge types:", "%d" % num_edge_types)

        ###########################################################
        #
        # Settings and placeholders
        #
        ###########################################################




        # Important -- Do not evaluate/print validation performance every iteration as it can take
        # substantial amount of time
        PRINT_PROGRESS_EVERY = 200

        print("Defining placeholders")
        placeholders = construct_placeholders(edge_types)



        print("Create minibatch iterator")


        ############################### Start MODEL TRAINING on CROSS VALIDATION #################################


        pos_df_all=pd.DataFrame()
        neg_df_all=pd.DataFrame()

        lr = FLAGS.learning_rate
        epochs = FLAGS.epochs
        batch_size = FLAGS.batch_size
        dr = FLAGS.dropout
        performace_metric_file_path = out_dir + 'run_' +str(run_) +'/'+ 'performance_metric' +'_drugfeat_'+\
                                      str(use_drug_feat_option)+'_e_'+str(epochs) +'_lr_'+str(lr) +'_batch_'+ str(batch_size) +'_dr_'+\
                                      str(dr)+'.txt'
        os.makedirs(os.path.dirname(performace_metric_file_path), exist_ok=True)
        performance_metric_file = open(performace_metric_file_path, 'w')

        for i in range(number_of_folds):
            os.makedirs(os.path.dirname(performace_metric_file_path), exist_ok=True)
            performance_metric_file = open(performace_metric_file_path, 'a')
            performance_metric_file.write('\nVALIDATION FOLD NO: '+str(i))
            performance_metric_file.write('\n\n')

            minibatch = EdgeMinibatchIterator(
                adj_mats=adj_mats_orig,
                feat=feat,
                edge_types=edge_types,
                pos_drug_drug_validation_all_folds = cross_validation_folds_pos_drug_drug_edges,
                neg_drug_drug_validation_all_folds =  cross_validation_folds_neg_drug_drug_edges,
                non_drug_drug_validation_all_folds = cross_validation_folds_non_drug_drug_edges,
                neg_non_drug_drug_validation_all_folds=neg_cross_validation_folds_non_drug_drug_edges,
                current_val_fold_no=i,
                batch_size=FLAGS.batch_size,
            )
            print("Create model")
            model = DecagonModel(
                placeholders=placeholders,
                num_feat=num_feat,
                nonzero_feat=nonzero_feat,
                edge_types=edge_types,
                decoders=edge_type2decoder

            )
            print("Create optimizer")
            with tf.name_scope('optimizer'):
                opt = DecagonOptimizer(
                    embeddings=model.embeddings,
                    latent_inters=model.latent_inters,
                    latent_varies=model.latent_varies,
                    degrees=degrees,
                    edge_types=edge_types,
                    edge_type2dim=edge_type2dim,
                    placeholders=placeholders,
                    batch_size=FLAGS.batch_size,
                    margin=FLAGS.max_margin
                )


            print("Initialize session")
            sess = tf.Session()
            sess.run(tf.global_variables_initializer())
            feed_dict = {}

            ###########################################################
            #
            # Train model
            #
            ###########################################################

            print("Train model")

            for epoch in range(FLAGS.epochs):
                t1 = time.time()
                minibatch.shuffle()
                itr = 0
                while not minibatch.end():
                    # Construct feed dictionary
                    feed_dict = minibatch.next_minibatch_feed_dict(placeholders=placeholders)
                    feed_dict = minibatch.update_feed_dict(
                        feed_dict=feed_dict,
                        dropout=FLAGS.dropout,
                        placeholders=placeholders)
                    # print('Minibatch current edge type:', minibatch.idx2edge_type[minibatch.current_edge_type_idx])


                    outs = sess.run([opt.opt_op, opt.cost, opt.batch_edge_type_idx], feed_dict=feed_dict)
                    train_cost = outs[1]
                    batch_edge_type = outs[2]

                    if itr % PRINT_PROGRESS_EVERY == 0:
                        model_predcition = get_model_prediction(minibatch.idx2edge_type[minibatch.current_edge_type_idx], feed_dict, placeholders, minibatch, \
                                                                sess, opt)
                        val_auc, val_auprc, val_apk = get_accuracy_scores(
                            minibatch.val_edges, minibatch.val_edges_false,
                            minibatch.idx2edge_type[minibatch.current_edge_type_idx], adj_mats_orig, model_predcition)

                        print("Epoch:", "%04d" % (epoch + 1), "Iter:", "%04d" % (itr + 1), "Edge:", "%04d" % batch_edge_type,
                              "train_loss=", "{:.5f}".format(train_cost),
                              "val_roc=", "{:.5f}".format(val_auc), "val_auprc=", "{:.5f}".format(val_auprc),
                              "val_apk=", "{:.5f}".format(val_apk))

                    itr += 1
                print("epoch time=", "{:.5f}".format(time.time() - t1))

            print("Optimization finished!")

            performance_metric_file.write('\n\nFinal result')
            for et in range(num_edge_types):
                model_predcition = get_model_prediction(minibatch.idx2edge_type[et], feed_dict, placeholders, minibatch,
                                                        sess, opt)
                roc_score, auprc_score, apk_score = get_accuracy_scores(
                    minibatch.val_edges, minibatch.val_edges_false, minibatch.idx2edge_type[et], adj_mats_orig, model_predcition)
                print("Edge type=", "[%02d, %02d, %02d]" % minibatch.idx2edge_type[et])
                print("Edge type:", "%04d" % et, "Val AUROC score", "{:.5f}".format(roc_score))
                print("Edge type:", "%04d" % et, "Val AUPRC score", "{:.5f}".format(auprc_score))
                print("Edge type:", "%04d" % et, "Val AP@k score", "{:.5f}".format(apk_score))
                print()

                if(minibatch.idx2edge_type[et][0]==1 & minibatch.idx2edge_type[et][1]==1):
                    cell_line_idx = minibatch.idx2edge_type[et][2]
                    cell_line = idx_2_cell_line[cell_line_idx]
                    performance_metric_file.write("\nEdge type: "+ str(minibatch.idx2edge_type[et])\
                    + "\ncell_line: " + cell_line + "\nval_roc = " + str(roc_score) + \
                                  "  val_auprc = " + str(auprc_score) +\
                                  "  val_apk= " + str(apk_score))

                    pos_df, neg_df = compute_drug_drug_link_probability(minibatch.val_edges[1,1][cell_line_idx],\
                                                                        minibatch.val_edges_false[1,1][cell_line_idx],cell_line,\
                                                                        model_predcition, i, idx_2_drug_node)
                    pos_df_all = pd.concat([pos_df_all, pos_df], axis=0)
                    neg_df_all = pd.concat([neg_df_all, neg_df], axis=0)

                else:
                    performance_metric_file.write("\nEdge type: " + str(minibatch.idx2edge_type[et]) \
                                  + "\nval_roc = " + str(roc_score) + "  val_auprc = " +\
                                  str(auprc_score) +"  val_apk = " + str(apk_score))




            performance_metric_file.close()

        save_drug_drug_link_probability(pos_df_all, neg_df_all, run_,use_drug_feat_option, FLAGS, out_dir)

        # output_df = preprocess_output_for_performance_evaluation(pos_df_all, neg_df_all)
    FLAGS.remove_flag_values(FLAGS.flag_values_dict())
    # return output_df