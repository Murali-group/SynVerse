from __future__ import division
from __future__ import print_function
from operator import itemgetter
from itertools import combinations
import time
import os

# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
# import tensorflow as tf

import numpy as np
import networkx as nx
import scipy.sparse as sp
from scipy.io import savemat, loadmat
from sklearn import metrics
import pandas as pd
import sys
import models.synverse.utils as utils
import models.synverse.cross_validation as cross_val
import random


import os.path as osp

import argparse

import torch.nn as nn
import sys
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, GAE, VGAE, SAGEConv, GMMConv, GATConv
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
from sklearn.decomposition import PCA
from tqdm import tqdm
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
import torchvision

import math
import random

from itertools import combinations, permutations, product

from torch.utils.tensorboard import SummaryWriter
from torch_geometric.utils import to_undirected, negative_sampling
import networkx as nx


import math
import random
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, precision_recall_curve
from torch_geometric.utils import to_undirected, negative_sampling
from torch_geometric.nn.models import InnerProductDecoder
from torch_geometric.nn.inits import reset


def parse_arguments():
    '''
    Initialize a parser and use it to parse the command line arguments
    :return: parsed dictionary of command line arguments
    '''
    parser = get_parser()
    opts = parser.parse_args()

    return opts


def sigmoid(x):
    return 1. / (1 + np.exp(-x))


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








# def construct_placeholders(edge_types):
#     placeholders = {
#         'batch': tf.placeholder(tf.int32, name='batch'),
#         'neg_batch': tf.placeholder(tf.int32, name='neg_batch'),
#         'batch_edge_type_idx': tf.placeholder(tf.int32, shape=(), name='batch_edge_type_idx'),
#         'batch_row_edge_type': tf.placeholder(tf.int32, shape=(), name='batch_row_edge_type'),
#         'batch_col_edge_type': tf.placeholder(tf.int32, shape=(), name='batch_col_edge_type'),
#         'degrees': tf.placeholder(tf.int32),
#         'dropout': tf.placeholder_with_default(0., shape=()),
#     }
#     placeholders.update({
#         'adj_mats_%d,%d,%d' % (i, j, k): tf.sparse_placeholder(tf.float32)
#         for i, j in edge_types for k in range(edge_types[i, j])})
#     placeholders.update({
#         'feat_%d' % i: tf.sparse_placeholder(tf.float32)
#         for i, _ in edge_types})
#     return placeholders


# def initial_model_setting(config_map):
#     flags = tf.app.flags
#     FLAGS = flags.FLAGS
#     decagon_settings = config_map['ml_models_settings']['algs']['decagon']
#     flags.DEFINE_integer('neg_sample_size', decagon_settings['neg_sample_size'], 'Negative sample size.')
#     flags.DEFINE_float('learning_rate', decagon_settings['learning_rate'], 'Initial learning rate.')
#     flags.DEFINE_integer('epochs', decagon_settings['epochs'], 'Number of epochs to train.')
#     flags.DEFINE_integer('hidden1', 64, 'Number of units in hidden layer 1.')
#     flags.DEFINE_integer('hidden2', 32, 'Number of units in hidden layer 2.')
#     flags.DEFINE_float('weight_decay', decagon_settings['weight_decay'], 'Weight for L2 loss on embedding matrix.')
#     flags.DEFINE_float('dropout', decagon_settings['dropout'], 'Dropout rate (1 - keep probability).')
#     flags.DEFINE_float('max_margin', decagon_settings['max_margin'], 'Max margin parameter in hinge loss')
#     flags.DEFINE_integer('batch_size', decagon_settings['batch_size'], 'minibatch size.')
#     flags.DEFINE_boolean('bias', decagon_settings['bias'], 'Bias term.')
#     return FLAGS

def run_synverse_model(ppi_sparse_matrix, gene_node_2_idx, drug_target_df, drug_maccs_keys_feature_df, synergy_df, non_synergy_df,\
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
    # drug_degrees_list = [np.array(drug_adj.sum(axis=0)).squeeze() for drug_adj in drug_drug_adj_list]

    print('finished network building')

    print('In Final network:\n Genes:%d Targetted Genes:%d  Drugs:%d' % (n_genes, tagetted_genes, n_drugs))


    # data representation
    adj_mats_init = {
        (0, 0): [gene_adj],
        (0, 1): [target_drug_adj],
        (1, 0): [drug_target_adj],
        (1, 1): drug_drug_adj_list,

    }
    # degrees = {
    #     0: [gene_degrees],
    #     1: drug_degrees_list,
    # }


    ###########################    CROSS VALIDATION PREPARATION    ######################################

    # cross validation folds contain only drug_pair index from synergy_df. Convert validation folds into list of (drug-idx, drug-idx, cell_line_idx) pairs.
    #after the following two processing both pos and neg cross validation folds will contain bot (x,y,cell_line) and (y,x,cell_line pairs.)
    edges_all_cell_line = list(zip(synergy_df['Drug1_idx'], synergy_df['Drug2_idx'], synergy_df['Cell_line_idx']))
    # print('all cell line  edges index length: ', len(edges_all_cell_line))
    temp_cross_validation_folds = {}
    for fold in cross_validation_folds_pos_drug_drug_edges:
        # print('test: ', fold, len(cross_validation_folds[fold]), cross_validation_folds[fold])
        temp_cross_validation_folds[fold] = [edges_all_cell_line[x] for x in cross_validation_folds_pos_drug_drug_edges[fold]]
        temp_cross_validation_folds[fold] += [(drug_2_idx,drug_1_idx,cell_line_idx) for drug_1_idx,drug_2_idx,cell_line_idx in  temp_cross_validation_folds[fold]]
        # print(temp_cross_validation_folds[fold][0:10], cross_validation_folds_pos_drug_drug_edges[fold][0:10])
    cross_validation_folds_pos_drug_drug_edges = temp_cross_validation_folds

    # cross validation folds contain only drug_pair index from non_synergy_df. Convert validation folds into list of (drug-idx, drug-idx) pairs.
    temp_cross_validation_folds = {}
    neg_edges_all_cell_line = list(zip(non_synergy_df['Drug1_idx'], non_synergy_df['Drug2_idx'], non_synergy_df['Cell_line_idx']))
    for fold in cross_validation_folds_neg_drug_drug_edges:
        # print('test: ', fold, len(cross_validation_folds[fold]), cross_validation_folds[fold])
        temp_cross_validation_folds[fold] = [neg_edges_all_cell_line[x] for x in cross_validation_folds_neg_drug_drug_edges[fold]]
        temp_cross_validation_folds[fold] += [(drug_2_idx, drug_1_idx, cell_line_idx) for drug_1_idx, drug_2_idx, cell_line_idx in
                                              temp_cross_validation_folds[fold]]
        # print(temp_cross_validation_folds[fold][0:10], cross_validation_folds_pos_drug_drug_edges[fold][0:10])
    cross_validation_folds_neg_drug_drug_edges = temp_cross_validation_folds

    non_drug_drug_edge_types = [(0,0),(0,1)]
    cross_validation_folds_non_drug_drug_edges = cross_val.create_cross_val_split_non_drug_drug_edges(number_of_folds, adj_mats_init,\
                                                                                            non_drug_drug_edge_types)
    neg_cross_validation_folds_non_drug_drug_edges = cross_val.create_neg_cross_val_split_non_drug_drug_edges(number_of_folds,
                                                                                            adj_mats_init, \
                                                                                            non_drug_drug_edge_types)


    ######################## NODE FEATURE MATRIX CREATION ###########################################

    # featureless (genes)
    gene_feat = sp.identity(n_genes)
    gene_nonzero_feat, gene_num_feat = gene_feat.shape
    gene_feat = utils.sparse_to_tuple(gene_feat.tocoo())

    # FLAGS = initial_model_setting(config_map)

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
            drug_feat = utils.sparse_to_tuple(drug_feat.tocoo())

        else:
        # #one hot encoding for drug features
            drug_feat = sp.identity(n_drugs)
            drug_nonzero_feat, drug_num_feat = drug_feat.shape
            drug_feat = utils.sparse_to_tuple(drug_feat.tocoo())

        #

class GAEwithK(torch.nn.Module):
    r"""The Graph Auto-Encoder model from the
    `"Variational Graph Auto-Encoders" <https://arxiv.org/abs/1611.07308>`_
    paper based on user-defined encoder and decoder models.

    Args:
        encoder (Module): The encoder module.
        decoder (Module, optional): The decoder module. If set to :obj:`None`,
            will default to the
            :class:`torch_geometric.nn.models.InnerProductDecoder`.
            (default: :obj:`None`)
    """

    def __init__(self, encoder, decoder=None):
        super(GAEwithK, self).__init__()
        self.encoder = encoder
        self.decoder = InnerProductDecoder() if decoder is None else decoder
        GAEwithK.reset_parameters(self)

    def reset_parameters(self):
        reset(self.encoder)
        reset(self.decoder)

    def encode(self, *args, **kwargs):
        r"""Runs the encoder and computes node-wise latent variables."""
        return self.encoder(*args, **kwargs)

    def decode(self, *args, **kwargs):
        r"""Runs the decoder and computes edge probabilties."""
        return self.decoder(*args, **kwargs)

    def recon_loss(self, z, pos_edge_index, neg_edge_index):
        r"""Given latent variables :obj:`z`, computes the binary cross
        entropy loss for positive edges :obj:`pos_edge_index` and negative
        sampled edges.

        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
            pos_edge_index (LongTensor): The positive edges to train against.
        """
        pos_loss = -torch.log(
            self.decoder(z, pos_edge_index, sigmoid=True) + EPS).mean()

        neg_loss = -torch.log(1 -
                              self.decoder(z, neg_edge_index, sigmoid=True) +
                              EPS).mean()

        return pos_loss + neg_loss

    def test(self, z, pos_edge_index, neg_edge_index):
        r"""Given latent variables :obj:`z`, positive edges
        :obj:`pos_edge_index` and negative edges :obj:`neg_edge_index`,
        computes area under the ROC curve (AUC) and average precision (AP)
        scores.

        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
            pos_edge_index (LongTensor): The positive edges to evaluate
                against.
            neg_edge_index (LongTensor): The negative edges to evaluate
                against.
        """
        pos_y = z.new_ones(pos_edge_index.size(1))
        neg_y = z.new_zeros(neg_edge_index.size(1))
        y = torch.cat([pos_y, neg_y], dim=0)

        pos_pred = self.decoder(z, pos_edge_index, sigmoid=True)
        neg_pred = self.decoder(z, neg_edge_index, sigmoid=True)

        pred = torch.cat([pos_pred, neg_pred], dim=0)

        y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()
        pr, rec, thresholds = precision_recall_curve(y, pred)
        pd.DataFrame([y, pred], index=['true', 'pred']).T.to_csv('preds.csv')
        pd.DataFrame([pr, rec, thresholds], index=['pr', 'rec', 'thres']).T.to_csv('pr.csv')
        return utils.precision_at_k(y, pred, pos_edge_index.size(1)), average_precision_score(y, pred)


class Encoder(torch.nn.Module):
    def __init__(self, h_sizes):
        super(Encoder, self).__init__()

        self.hidden = nn.ModuleList()
        self.num_hidden = len(h_sizes) - 1
        for k in range(self.num_hidden):
            self.hidden.append(GCNConv(h_sizes[k], h_sizes[k + 1], cached=False))

    def forward(self, x, edge_index):
        for k in range(self.num_hidden):
            # x = F.relu(F.dropout(self.hidden[k](x, edge_index), p=0.2, training=self.training))
            x = F.relu(self.hidden[k](x, edge_index))
            x = F.dropout(x, p=0.2, training=self.training)
        return x


class TFDecoder(torch.nn.Module):
    def __init__(self, num_nodes, TFIDs):
        super(TFDecoder, self).__init__()
        self.TFIDs = list(TFIDs)
        self.num_nodes = num_nodes
        self.in_dim = 1  # one relation type
        # self.weight = nn.Parameter(torch.Tensor(len(self.TFIDs)))
        self.weight = nn.Parameter(torch.Tensor(self.num_nodes))
        self.reset_parameters()

    def forward(self, z, edge_index, sigmoid=True):
        # newWeight = torch.zeros(self.num_nodes).to(dev)
        # nCnt = 0
        # for idx in self.TFIDs:
        #    newWeight[idx] = self.weight[nCnt]
        #    nCnt += 1
        zNew = torch.mul(z.t(), self.weight).t()
        # print(self.weight[0])#, newWeight[850])
        # sys.exit()
        value = (zNew[edge_index[0]] * z[edge_index[1]]).sum(dim=1)

        return torch.sigmoid(value) if sigmoid else value

    def reset_parameters(self):
        self.weight.data.normal_(std=1 / np.sqrt(self.in_dim))


class RESCALDecoder(torch.nn.Module):
    def __init__(self, out_dim):
        super(RESCALDecoder, self).__init__()
        self.out_dim = out_dim
        self.in_dim = 1  # one relation type
        self.weight = nn.Parameter(torch.Tensor(self.out_dim, self.out_dim))
        self.reset_parameters()

    def forward(self, z, edge_index, sigmoid=True):
        # zNew = z.clone()*self.weight
        zNew = torch.matmul(z.clone(), self.weight)
        # zNew = z*self.weight
        # print(edge_index)
        # print(zNew.shape,self.weight)
        value = (zNew[edge_index[0]] * z[edge_index[1]]).sum(dim=1)

        # self.weight[edge_index[0]]
        # print(value, edge_index)
        # value = (1 * z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
        # print(edge_index.shape,value.shape)
        return torch.sigmoid(value) if sigmoid else value

    def reset_parameters(self):
        self.weight.data.normal_(std=1 / np.sqrt(self.in_dim))


def train():
    model.train()
    optimizer.zero_grad()
    z = model.encode(x, train_pos_edge_index)
    loss = model.recon_loss(z, train_pos_only_edge_index, train_neg_edge_index)
    loss.backward()
    optimizer.step()
    return (loss)


def test(pos_edge_index, neg_edge_index):
    model.eval()
    with torch.no_grad():
        z = model.encode(x, train_pos_edge_index)
    epr, ap = model.test(z, pos_edge_index, neg_edge_index)
    return z, epr, ap


def val(pos_edge_index, neg_edge_index):
    model.eval()
    with torch.no_grad():
        z = model.encode(x, train_pos_edge_index)
    loss = model.recon_loss(z, pos_edge_index, neg_edge_index)
    return loss
