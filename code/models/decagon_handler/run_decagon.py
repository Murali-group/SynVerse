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

# sys.path.insert(1, '/home/tasnina/Projects/Synverse')
print(sys.path)
from models.decagon_handler.decagon_minibatch import EdgeMinibatchIterator
from submodules.decagon.decagon.deep.optimizer import DecagonOptimizer
from submodules.decagon.decagon.deep.model import DecagonModel
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

def predict_links( edge_type, feed_dict, placeholders, minibatch, sess, opt):
    feed_dict.update({placeholders['dropout']: 0})
    feed_dict.update({placeholders['batch_edge_type_idx']: minibatch.edge_type2idx[edge_type]})
    feed_dict.update({placeholders['batch_row_edge_type']: edge_type[0]})
    feed_dict.update({placeholders['batch_col_edge_type']: edge_type[1]})
    rec = sess.run(opt.predictions, feed_dict=feed_dict)
    return rec

def compute_link_probability(edges_pos, edges_neg,edge_type,rec, val_fold, idx_2_drug_node, idx_2_cell_line):
    #if we pass edge_type(1,1,x) then it will give score for drug-drug-links
    if(edge_type[0]==1 & edge_type[1]==1):
        pos_edge_dict = {'drug_1_idx': [], 'drug_2_idx': [], 'cell_line_idx':[], 'model_score': [], 'probability': [], 'val_fold': []}
        neg_edge_dict = {'drug_1_idx': [], 'drug_2_idx': [], 'cell_line_idx':[], 'model_score': [], 'probability': [], 'val_fold': []}
        for u, v in edges_pos[edge_type[:2]][edge_type[2]]:
            pos_edge_dict['drug_1_idx'].append(u)
            pos_edge_dict['drug_2_idx'].append(v)
            pos_edge_dict['cell_line_idx'].append(edge_type[2])
            pos_edge_dict['model_score'].append(rec[u,v])
            pos_edge_dict['probability'].append(sigmoid(rec[u,v]))
            pos_edge_dict['val_fold'].append(val_fold)

        for u, v in edges_neg[edge_type[:2]][edge_type[2]]:
            neg_edge_dict['drug_1_idx'].append(u)
            neg_edge_dict['drug_2_idx'].append(v)
            neg_edge_dict['cell_line_idx'].append(edge_type[2])
            neg_edge_dict['model_score'].append(rec[u, v])
            neg_edge_dict['probability'].append(sigmoid(rec[u, v]))
            neg_edge_dict['val_fold'].append(val_fold)

        pos_df = pd.DataFrame.from_dict(pos_edge_dict)
        neg_df = pd.DataFrame.from_dict(neg_edge_dict)

        pos_df['drug_1'] = pos_df['drug_1_idx'].apply(lambda x: idx_2_drug_node[x])
        pos_df['drug_2'] = pos_df['drug_2_idx'].apply(lambda x: idx_2_drug_node[x])
        pos_df['cell_line'] = pos_df['cell_line_idx'].apply(lambda x: idx_2_cell_line[x])


        neg_df['drug_1'] = neg_df['drug_1_idx'].apply(lambda x: idx_2_drug_node[x])
        neg_df['drug_2'] = neg_df['drug_2_idx'].apply(lambda x: idx_2_drug_node[x])
        neg_df['cell_line'] = neg_df['cell_line_idx'].apply(lambda x: idx_2_cell_line[x])

        pos_df = pos_df[['drug_1','drug_2','cell_line', 'model_score','probability','val_fold']].sort_values(by=['probability'],\
                                                                                                 ascending=False)
        neg_df = neg_df[['drug_1', 'drug_2','cell_line', 'model_score', 'probability', 'val_fold']].sort_values(by=['probability'],\
                                                                                                    ascending=False)
        return pos_df, neg_df
def save_link_probability(pos_df, neg_df, out_dir):
    #inputs: df with link prediction probability after applying sigmoid on models predicted score for an edges(positive, negative)
    pos_out_file = out_dir + 'positive_validation_edge_scores.tsv'
    neg_out_file = out_dir + 'negative_validation_edge_scores.tsv'

    os.makedirs(os.path.dirname(pos_out_file), exist_ok=True)
    os.makedirs(os.path.dirname(neg_out_file), exist_ok=True)

    pos_df.to_csv(pos_out_file, sep='\t')
    neg_df.to_csv(neg_out_file, sep='\t')

def get_accuracy_scores(edges_pos, edges_neg, edge_type, adj_mats_orig, rec):

    # rec = predict_links(edge_type, feed_dict, placeholders, minibatch, sess, opt)
    print('edge type in get_accuracy_scores:', edge_type)
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
        print('all (x,y)  and (y,x) tuple present')
        return 'undirected'
    else:
        print('Not present')
        return 'directed'

def run_decagon_model(ppi_sparse_matrix,gene_node_2_idx,drug_target_df,synergy_df,\
                              cross_validation_folds, config_map):

    out_dir = config_map['project_dir']+ config_map['output_dir'] + 'decagon/'

    gene_adj = nx.adjacency_matrix(nx.convert_matrix.from_scipy_sparse_matrix(ppi_sparse_matrix, create_using=nx.Graph(),edge_attribute=None))
    gene_degrees = np.array(gene_adj.sum(axis=0)).squeeze()
    genes_in_ppi = gene_node_2_idx.keys()

    synergistics_drugs = set(list(synergy_df['Drug1_pubchem_cid'])).union(set(list(synergy_df['Drug2_pubchem_cid'])))
    print('number of drugs after applying threshold on synergy data:', len(synergistics_drugs))
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

    # cross validation folds contain only drug_pair index from synergy_df. Conver validation folds into list of (drug-idx, drug-idx) pairs.
    edges_all_cell_line = list(zip(synergy_df['Drug1_idx'], synergy_df['Drug2_idx'], synergy_df['Cell_line_idx']))
    print('all cell line  edges index length: ', len(edges_all_cell_line))
    temp_cross_validation_folds = {}
    for fold in cross_validation_folds:
        # print('test: ', fold, len(cross_validation_folds[fold]), cross_validation_folds[fold])
        temp_cross_validation_folds[fold] = [edges_all_cell_line[x] for x in cross_validation_folds[fold]]
        print(temp_cross_validation_folds[fold][0:10],cross_validation_folds[fold][0:10])
    cross_validation_folds = temp_cross_validation_folds



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

    # featureless (genes)
    gene_feat = sp.identity(n_genes)
    gene_nonzero_feat, gene_num_feat = gene_feat.shape
    gene_feat = preprocessing.sparse_to_tuple(gene_feat.tocoo())

    # features (drugs)
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
    print("Edge types:", "%d" % num_edge_types)

    ###########################################################
    #
    # Settings and placeholders
    #
    ###########################################################

    FLAGS = initial_model_setting(config_map)
    # Important -- Do not evaluate/print validation performance every iteration as it can take
    # substantial amount of time
    PRINT_PROGRESS_EVERY = 150

    print("Defining placeholders")
    placeholders = construct_placeholders(edge_types)


    ###########################################################
    #
    # Create minibatch iterator, model and optimizer
    #
    ###########################################################

    print("Create minibatch iterator")

    pos_df_all=pd.DataFrame()
    neg_df_all=pd.DataFrame()

    performace_metric_file_path = out_dir + 'performance_metric.txt'
    for validation_fold in cross_validation_folds:
        os.makedirs(os.path.dirname(performace_metric_file_path), exist_ok=True)
        performance_metric_file = open(performace_metric_file_path, 'a')
        performance_metric_file.write('\nVALIDATION FOLD NO: '+str(validation_fold))
        performance_metric_file.write('\n\n')
        minibatch = EdgeMinibatchIterator(
            adj_mats=adj_mats_orig,
            feat=feat,
            edge_types=edge_types,
            validation_fold=cross_validation_folds[validation_fold],
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

                t = time.time()


                outs = sess.run([opt.opt_op, opt.cost, opt.batch_edge_type_idx], feed_dict=feed_dict)
                train_cost = outs[1]
                batch_edge_type = outs[2]

                if itr % PRINT_PROGRESS_EVERY == 0:
                    model_predcition = predict_links( minibatch.idx2edge_type[minibatch.current_edge_type_idx], feed_dict, placeholders, minibatch,\
                                                     sess, opt)
                    val_auc, val_auprc, val_apk = get_accuracy_scores(
                        minibatch.val_edges, minibatch.val_edges_false,
                        minibatch.idx2edge_type[minibatch.current_edge_type_idx], adj_mats_orig, model_predcition)

                    print("Epoch:", "%04d" % (epoch + 1), "Iter:", "%04d" % (itr + 1), "Edge:", "%04d" % batch_edge_type,
                          "train_loss=", "{:.5f}".format(train_cost),
                          "val_roc=", "{:.5f}".format(val_auc), "val_auprc=", "{:.5f}".format(val_auprc),
                          "val_apk=", "{:.5f}".format(val_apk), "time=", "{:.5f}".format(time.time() - t))
                    # try:
                    #     outfile.write("Epoch:" + str((epoch + 1)) +"\nIter:"+str((itr + 1)) + "\nEdge:"+ str (batch_edge_type)+
                    #           "\ntrain_loss=" +str((train_cost))+
                    #           "val_roc=" + str(val_auc) + "val_auprc="+str(val_auprc)+
                    #           "val_apk="+str(val_apk)+"time="+str((time.time() - t)))
                    # except:
                    #     print('error writing in file')




                itr += 1

        print("Optimization finished!")

        performance_metric_file.write('\n\nFinal result')
        for et in range(num_edge_types):
            model_predcition = predict_links(minibatch.idx2edge_type[et], feed_dict, placeholders, minibatch,\
                                             sess, opt)
            roc_score, auprc_score, apk_score = get_accuracy_scores(
                minibatch.val_edges, minibatch.val_edges_false, minibatch.idx2edge_type[et], adj_mats_orig, model_predcition)
            print("Edge type=", "[%02d, %02d, %02d]" % minibatch.idx2edge_type[et])
            print("Edge type:", "%04d" % et, "Val AUROC score", "{:.5f}".format(roc_score))
            print("Edge type:", "%04d" % et, "Val AUPRC score", "{:.5f}".format(auprc_score))
            print("Edge type:", "%04d" % et, "Val AP@k score", "{:.5f}".format(apk_score))
            print()

            if(minibatch.idx2edge_type[et][0]==1 & minibatch.idx2edge_type[et][1]==1):
                cell_line = idx_2_cell_line[minibatch.idx2edge_type[et][2]]
                performance_metric_file.write("\nEdge type: "+ str(minibatch.idx2edge_type[et])\
                + "\ncell_line: " + cell_line + "\nval_roc = " + str(roc_score) + \
                              "  val_auprc = " + str(auprc_score) +\
                              "  val_apk= " + str(apk_score))

                pos_df, neg_df = compute_link_probability(minibatch.val_edges, minibatch.val_edges_false, \
                                                          minibatch.idx2edge_type[et], \
                                                          model_predcition, validation_fold, idx_2_drug_node,
                                                          idx_2_cell_line)
                pos_df_all = pd.concat([pos_df_all, pos_df], axis=0)
                neg_df_all = pd.concat([neg_df_all, neg_df], axis=0)

            else:
                performance_metric_file.write("\nEdge type: " + str(minibatch.idx2edge_type[et]) \
                              + "\nval_roc = " + str(roc_score) + "  val_auprc = " +\
                              str(auprc_score) +"  val_apk = " + str(apk_score))




        performance_metric_file.close()

    save_link_probability(pos_df_all, neg_df_all, out_dir)
