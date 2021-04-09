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
import models.synverse.minibatch as minibatch
from minibatch import MinibatchHandler
import random
import copy

from collections import defaultdict


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


EPS = 1e-15
MAX_LOGVAR = 10




# def parse_arguments():
#     '''
#     Initialize a parser and use it to parse the command line arguments
#     :return: parsed dictionary of command line arguments
#     '''
#     parser = get_parser()
#     opts = parser.parse_args()
#
#     return opts


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





class SynverseModel(torch.nn.Module):
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

    def __init__(self, encoder, decoders):
        super(SynverseModel, self).__init__()
        self.encoder = encoder
        self.decoders = decoders
        SynverseModel.reset_parameters(self)

    def reset_parameters(self):
        reset(self.encoder)
        for edge_type in self.decoders:
            reset(self.decoders[edge_type])

    # def encode(self, *args, **kwargs):
    #     r"""Runs the encoder and computes node-wise latent variables."""
    #     return self.encoder(*args, **kwargs)

    def encode(self):
        r"""Runs the encoder and computes node-wise latent variables."""
        return self.encoder()

    # def decode(self, *args, **kwargs):
    #     r"""Runs the decoder and computes edge probabilties."""
    #     return self.decoder(*args, **kwargs)

    def recon_loss(self, z, batch_pos_edge_index, batch_neg_edge_index, edge_type, edge_sub_type_idx):
        r"""Given latent variables :obj:`z`, computes the binary cross
        entropy loss for positive edges :obj:`pos_edge_index` and negative
        sampled edges.

        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
            pos_edge_index (LongTensor): The positive edges to train against.
        """
        pos_loss = -torch.log(
            self.decoders[edge_type](z, batch_pos_edge_index, edge_sub_type_idx, sigmoid=True) + EPS).mean()

        neg_loss = -torch.log(1 -
                              self.decoders[edge_type](z, batch_neg_edge_index, edge_sub_type_idx, sigmoid=True) +
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
    #h_sizes is an array. containing the gardual node number decrease from initial hidden_layer output  to final output_dim.\
    # Input dim is not included
    def __init__(self, h_sizes, node_feat_dict, train_pos_edges_dict,  edge_types):
        super(Encoder, self).__init__()

        self.edge_types = edge_types
        self.node_feat_dict = node_feat_dict
        self.train_pos_edges_dict = train_pos_edges_dict
        self.h_sizes = h_sizes
        self.num_hidden = len(h_sizes)
        self.hidden={} #dict of dict. self.hidden[0]=> contains a dictionary for first hidden layer. In this dictionary, keys = edge_type,
        # value = GCN layer dedicated to it.
        for hid_layer_no in range(self.num_hidden):
            self.hidden[hid_layer_no] = {}
            for edge_type in edge_types:
                if hid_layer_no == 0:
                    if edge_type in ['drug_drug', 'drug_target']:
                        input_feat_dim = node_feat_dict['drug'].size()[1]
                    else:
                        input_feat_dim = node_feat_dict['gene'].size()[1]

                    self.hidden[hid_layer_no][edge_type] = \
                        (EdgeTypeSpecGCNLayer(input_feat_dim, h_sizes[hid_layer_no], edge_type))
                else:
                    self.hidden[hid_layer_no][edge_type] = \
                        (EdgeTypeSpecGCNLayer(h_sizes[hid_layer_no-1], h_sizes[hid_layer_no], edge_type))


    def forward(self):
        # for one input layer
        # this will contain the output  i.e. embedding after hidden layer 1
        hidden_output = {}
        layer_input = copy.deepcopy(self.node_feat_dict)
        for hid_layer_no in range(self.num_hidden):
            hidden_output[hid_layer_no] = {}
            # hidden_output[1]={'drug': final_drug_embedding_from_hidden_layer_1, 'gene':final_gene_embedding_from_hidden_layer_1}
            # self.hidden1_output['drug'].append()
            for edge_type in self.edge_types:
                if edge_type in ['drug_drug', 'target_drug']:
                    #target_node = drug in both type of edges i.e.
                    # drug embedding is being computed. Hence, put under 'drug' key.
                    self.hidden_output[hid_layer_no]['drug'].append(
                        self.hidden[hid_layer_no][edge_type](layer_input, self.train_pos_edges_dict, edge_type))
                else:
                    self.hidden_output[hid_layer_no]['gene'].append(
                        self.hidden[hid_layer_no][edge_type](layer_input, self.train_pos_edges_dict,
                                                             edge_type))

            for node_type in self.hidden_output[hid_layer_no]:
                output = 0
                for i in self.hidden_output[hid_layer_no][node_type]:
                    output+=self.hidden_output[hid_layer_no][node_type][i]
                self.hidden_output[hid_layer_no][node_type] = output

            layer_input = hidden_output[hid_layer_no]

        return layer_input





class EdgeTypeSpecGCNLayer(torch.nn.Module):
    def __init__(self, in_channel, out_channel, edge_type):
        super(EdgeTypeSpecGCNLayer, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        if edge_type in ['gene_gene','drug_drug']:
            self.gcn = GCNConv(self.in_channel,self.out_channel, add_self_loops=True, cached=False)
        else:
            self.gcn = GCNConv(self.in_channel, self.out_channel, add_self_loops=False, normalize = False , cached=False)

    def forward(self, node_feat_dict, train_pos_edges_dict, edge_type):
        if edge_type in ['drug_drug', 'drug_target']: #source node = 'drug' in both type of edges. hence take 'drug_feat' in x
            x = node_feat_dict['drug']
        else:
            x = node_feat_dict['gene']
        adj_mat_list = train_pos_edges_dict[edge_type]

        output = 0
        for adj_mat in adj_mat_list:
            x = F.dropout(x, p=0.2, training = True)
            x = F.relu(self.gcn(x, adj_mat))
            output += x
        output = F.normalize(output, dim=1, p=2) #p=2 means l2-normalization
        return output

class BilinearDecoder(torch.nn.Module): #one decoder object for one edge_type
    def __init__(self, edge_type, n_sub_types, w_dim ):
        # n_sub_types = how many different weight matrix is needed to initialize
        # w_dim = wight matrix dimension will be w_dim * w_dim i.e. dimension of the final nodel embedding
        super(BilinearDecoder, self).__init__()
        self.edge_type = edge_type
        self.weights = []
        for i in range(n_sub_types):
            self.weights.append(utils.weight_matrix_glorot(w_dim, w_dim))

    def forward(self, z, batch_edges, edge_sub_type_idx, sigmoid=True):
        if self.edge_type == 'gene_gene':
            z1 = z['gene']
            z2 = z['gene']
        elif self.edge_type == 'target_drug':
            z1 = z['gene']
            z2 = z['drug']
        elif self.edge_type == 'drug-target':
            z1 = z['drug']
            z2 = z['target']
        else:
            z1 = z['drug']
            z2 = z['drug']

        row_embed = z1[batch_edges[0]]
        col_embed = z2[batch_edges[1]]
        product_1 = torch.mul(row_embed, self.weights[edge_sub_type_idx])
        # print(self.weight[0])#, newWeight[850])
        # sys.exit()
        score = (product_1 * col_embed.t())

        return torch.sigmoid(score) if sigmoid else score

    # def reset_parameters(self):
    #     self.weight.data.normal_(std=1 / np.sqrt(self.in_dim))
    #


class DedicomDecoder(torch.nn.Module): #one decoder object for one edge_type
    def __init__(self, edge_type, n_sub_types, w_dim ):
        # n_sub_types = how many different weight matrix is needed to initialize
        # w_dim = wight matrix dimension will be w_dim * w_dim i.e. dimension of the final nodel embedding
        super(DedicomDecoder, self).__init__()
        self.edge_type = edge_type
        self.global_weight = utils.weight_matrix_glorot(w_dim, w_dim)
        self.local_weights = []

        for i in range(n_sub_types):
            diagonal_vals = torch.reshape(utils.weight_matrix_glorot(w_dim, 1), [-1])
            self.local_weights.append(torch.diag(diagonal_vals))

    def forward(self, z, batch_edges, edge_sub_type_idx, sigmoid=True):
        if self.edge_type == 'gene_gene':
            z1 = z['gene']
            z2 = z['gene']
        elif self.edge_type == 'target_drug':
            z1 = z['gene']
            z2 = z['drug']
        elif self.edge_type == 'drug-target':
            z1 = z['drug']
            z2 = z['target']
        else:
            z1 = z['drug']
            z2 = z['drug']
        row_embed = z1[batch_edges[0]]
        col_embed = z2[batch_edges[1]]
        product_1 = torch.mul(row_embed, self.local_weights[edge_sub_type_idx])
        product_2 = torch.mul(product_1, self.global_weight)
        product_3 = torch.mul(product_2, self.local_weights[edge_sub_type_idx])
        score = product_3 * col_embed.t()

        return torch.sigmoid(score) if sigmoid else score

    # def reset_parameters(self):
    #     self.weight.data.normal_(std=1 / np.sqrt(self.in_dim))

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

#
def train(model, optimizer,  batch_pos_train_edges, batch_neg_train_edges, edge_type, edge_sub_type_idx ):
    model.train()
    optimizer.zero_grad()
    z = model.encode()
    loss = model.recon_loss(z, batch_pos_train_edges, batch_neg_train_edges, edge_type, edge_sub_type_idx)
    loss.backward()
    optimizer.step()
    return (loss)

#
# def test(model, pos_edge_index, neg_edge_index, edge_type, edge_sub_type_idx):
#     model.eval()
#     with torch.no_grad():
#         z = model.encode()
#     epr, ap = model.test(z, pos_edge_index, neg_edge_index)
#     return z, epr, ap
#
#
def val(model, pos_edge_index, neg_edge_index, edge_type, edge_sub_type_idx):
    model.eval()
    with torch.no_grad():
        z = model.encode()
    loss = model.recon_loss(z, pos_edge_index, neg_edge_index, edge_type, edge_sub_type_idx)
    return loss



def prepare_drug_feat(drug_maccs_keys_feature_df, drug_node_2_idx, n_drugs, use_drug_feat_option):
    if use_drug_feat_option:
        drug_maccs_keys_feature_df['drug_idx'] = drug_maccs_keys_feature_df['pubchem_cid']. \
            apply(lambda x: drug_node_2_idx[x])
        drug_maccs_keys_feature_df = drug_maccs_keys_feature_df.sort_values(by=['drug_idx'])
        assert len(drug_maccs_keys_feature_df) == n_drugs, 'problem in drug feat creation'
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

        return drug_feat

def get_set_containing_all_folds(cross_validation_folds_pos_drug_drug_edges):
    all_folds_edges = set()
    for fold,edges in cross_validation_folds_pos_drug_drug_edges.items():
        all_folds_edges = all_folds_edges.union(set(edges))
    return all_folds_edges

def prepare_train_edges(cross_validation_folds_pos_drug_drug_edges, cross_validation_folds_neg_drug_drug_edges, \
                    cross_validation_folds_pos_non_drug_drug_edges, cross_validation_folds_neg_non_drug_drug_edges,
                    fold_no, total_cell_lines):
    #this returns four dictionaries. one dict for each of train_pos, train_neg, val_pos, val_neg.
    # in each dictionary: key = edge_type i.e. ( 'gene_gene', 'target_drug', 'drug_target', 'drug_drug')
    # each key maps to a list of torchtensor. e.g. drug_drug key maps to a list of torch tensor where each tensor is for a cell line


    edge_types = ['gene_gene', 'target_drug', 'drug_target', 'drug_drug']
    train_pos_edges_dict = {edge_type: [] for edge_type in edge_types}
    train_neg_edges_dict = {edge_type: [] for edge_type in edge_types}
    val_pos_edges_dict = {edge_type: [] for edge_type in edge_types}
    val_neg_edges_dict = {edge_type: [] for edge_type in edge_types}


    # set drug drug edges
    val_pos_drug_drug_edges_set = set(cross_validation_folds_pos_drug_drug_edges[fold_no])
    all_folds_pos_edges = get_set_containing_all_folds(cross_validation_folds_pos_drug_drug_edges)
    train_pos_drug_drug_edges_set = all_folds_pos_edges.difference(val_pos_drug_drug_edges_set)

    val_neg_drug_drug_edges_set = set(cross_validation_folds_neg_drug_drug_edges[fold_no])
    all_folds_neg_edges = get_set_containing_all_folds(cross_validation_folds_neg_drug_drug_edges)
    train_neg_drug_drug_edges_set = all_folds_neg_edges.difference(val_neg_drug_drug_edges_set)

    for cell_line_idx in range(total_cell_lines):
        train_pos_source_nodes = [d1 for d1,d2,c in train_pos_drug_drug_edges_set if c==cell_line_idx]
        train_pos_target_nodes = [d2 for d1, d2, c in train_pos_drug_drug_edges_set if c == cell_line_idx]
        train_pos_edges = torch.stack([torch.LongTensor(train_pos_source_nodes),torch.LongTensor(train_pos_target_nodes)],dim=0)

        train_neg_source_nodes = [d1 for d1, d2, c in train_neg_drug_drug_edges_set if c == cell_line_idx]
        train_neg_target_nodes = [d2 for d1, d2, c in train_neg_drug_drug_edges_set if c == cell_line_idx]
        train_neg_edges = torch.stack([torch.LongTensor(train_neg_source_nodes), torch.LongTensor(train_neg_target_nodes)], dim=0)

        val_pos_source_nodes = [d1 for d1, d2, c in val_pos_drug_drug_edges_set if c == cell_line_idx]
        val_pos_target_nodes = [d2 for d1, d2, c in val_pos_drug_drug_edges_set if c == cell_line_idx]
        val_pos_edges = torch.stack(
            [torch.LongTensor(val_pos_source_nodes), torch.LongTensor(val_pos_target_nodes)], dim=0)

        val_neg_source_nodes = [d1 for d1, d2, c in val_neg_drug_drug_edges_set if c == cell_line_idx]
        val_neg_target_nodes = [d2 for d1, d2, c in val_neg_drug_drug_edges_set if c == cell_line_idx]
        val_neg_edges = torch.stack(
            [torch.LongTensor(val_neg_source_nodes), torch.LongTensor(val_neg_target_nodes)], dim=0)

        train_pos_edges_dict['drug_drug'].append(train_pos_edges)
        train_neg_edges_dict['drug_drug'].append(train_neg_edges)
        val_pos_edges_dict['drug_drug'].append(val_pos_edges)
        val_neg_edges_dict['drug_drug'].append(val_neg_edges)

    #gene_gene edges, target_drug_edges, drug_target_edges

    non_drug_drug_edge_types = ['gene_gene', 'target_drug', 'drug_target']

    for edge_type in non_drug_drug_edge_types:

        val_pos_non_drug_drug_edges_set = set(cross_validation_folds_pos_non_drug_drug_edges[edge_type][fold_no])
        all_folds_pos_edges = get_set_containing_all_folds(cross_validation_folds_pos_non_drug_drug_edges[edge_type])
        train_pos_non_drug_drug_edges_set = all_folds_pos_edges.difference(val_pos_non_drug_drug_edges_set)

        val_neg_non_drug_drug_edges_set = set(cross_validation_folds_neg_non_drug_drug_edges[edge_type][fold_no])
        all_folds_neg_edges = get_set_containing_all_folds(cross_validation_folds_neg_non_drug_drug_edges[edge_type])
        train_neg_non_drug_drug_edges_set = all_folds_neg_edges.difference(val_neg_non_drug_drug_edges_set)


        train_pos_source_nodes = [d1 for d1, d2 in train_pos_non_drug_drug_edges_set]
        train_pos_target_nodes = [d2 for d1, d2 in train_pos_non_drug_drug_edges_set]
        train_pos_edges = torch.stack(
            [torch.LongTensor(train_pos_source_nodes), torch.LongTensor(train_pos_target_nodes)], dim=0)

        train_neg_source_nodes = [d1 for d1, d2 in train_neg_non_drug_drug_edges_set]
        train_neg_target_nodes = [d2 for d1, d2 in train_neg_non_drug_drug_edges_set]
        train_neg_edges = torch.stack(
            [torch.LongTensor(train_neg_source_nodes), torch.LongTensor(train_neg_target_nodes)], dim=0)

        val_pos_source_nodes = [d1 for d1, d2 in val_pos_non_drug_drug_edges_set]
        val_pos_target_nodes = [d2 for d1, d2 in val_pos_non_drug_drug_edges_set]
        val_pos_edges = torch.stack(
            [torch.LongTensor(val_pos_source_nodes), torch.LongTensor(val_pos_target_nodes)], dim=0)

        val_neg_source_nodes = [d1 for d1, d2 in val_neg_non_drug_drug_edges_set]
        val_neg_target_nodes = [d2 for d1, d2  in val_neg_non_drug_drug_edges_set]
        val_neg_edges = torch.stack(
            [torch.LongTensor(val_neg_source_nodes), torch.LongTensor(val_neg_target_nodes)], dim=0)

        train_pos_edges_dict[edge_type].append(train_pos_edges)
        train_neg_edges_dict[edge_type].append(train_neg_edges)
        val_pos_edges_dict[edge_type].append(val_pos_edges)
        val_neg_edges_dict[edge_type].append(val_neg_edges)


    return  train_pos_edges_dict, train_neg_edges_dict, val_pos_edges_dict, val_neg_edges_dict




def run_synverse_model(ppi_sparse_matrix, gene_node_2_idx, drug_target_df, drug_maccs_keys_feature_df, synergy_df, non_synergy_df,\
                      cross_validation_folds_pos_drug_drug_edges, cross_validation_folds_neg_drug_drug_edges,\
                      run_, out_dir, config_map):

    #model setup
    synverse_settings = config_map['ml_models_settings']['algs']['synverse']
    learning_rate =  synverse_settings['learning_rate']
    epochs = synverse_settings['epochs']
    hidden1 = 64
    hidden2 = 32
    weight_decay = synverse_settings['weight_decay']
    dropout = synverse_settings['dropout']
    max_margin = synverse_settings['max_margin']
    batch_size = synverse_settings['batch_size']
    bias = synverse_settings['bias']

    neg_fact = config_map['ml_models_settings']['cross_val']['neg_fact']

    #
    number_of_folds = config_map['ml_models_settings']['cross_val']['folds']

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
    edge_types = ['gene_gene', 'target_drug', 'drug_target', 'drug_drug']
    adj_mats_init = {}
    adj_mats_init['gene_gene'] = [gene_adj]
    adj_mats_init['target_drug'] = [target_drug_adj]
    adj_mats_init['drug_target'] = [drug_target_adj]
    adj_mats_init['drug_drug'] = drug_drug_adj_list



        ###########################    CROSS VALIDATION PREPARATION    ######################################

    # cross validation folds contain only drug_pair index from synergy_df. Convert validation folds into list of (drug-idx, drug-idx, cell_line_idx) pairs.
    # after the following two processing both pos and neg cross validation folds will contain both (x,y,cell_line) and (y,x,cell_line tuples.)
    edges_all_cell_line = list(zip(synergy_df['Drug1_idx'], synergy_df['Drug2_idx'], synergy_df['Cell_line_idx']))
    # print('all cell line  edges index length: ', len(edges_all_cell_line))
    temp_cross_validation_folds = {}
    for fold in cross_validation_folds_pos_drug_drug_edges:
        # print('test: ', fold, len(cross_validation_folds[fold]), cross_validation_folds[fold])
        temp_cross_validation_folds[fold] = [edges_all_cell_line[x] for x in
                                             cross_validation_folds_pos_drug_drug_edges[fold]]
        temp_cross_validation_folds[fold] += [(drug_2_idx, drug_1_idx, cell_line_idx) for
                                              drug_1_idx, drug_2_idx, cell_line_idx in
                                              temp_cross_validation_folds[fold]]
        # print(temp_cross_validation_folds[fold][0:10], cross_validation_folds_pos_drug_drug_edges[fold][0:10])
    cross_validation_folds_pos_drug_drug_edges = temp_cross_validation_folds


    # cross validation folds contain only drug_pair index from non_synergy_df. Convert validation folds into list of (drug-idx, drug-idx) pairs.
    temp_cross_validation_folds = {}
    neg_edges_all_cell_line = list(
        zip(non_synergy_df['Drug1_idx'], non_synergy_df['Drug2_idx'], non_synergy_df['Cell_line_idx']))
    for fold in cross_validation_folds_neg_drug_drug_edges:
        # print('test: ', fold, len(cross_validation_folds[fold]), cross_validation_folds[fold])
        temp_cross_validation_folds[fold] = [neg_edges_all_cell_line[x] for x in
                                             cross_validation_folds_neg_drug_drug_edges[fold]]
        temp_cross_validation_folds[fold] += [(drug_2_idx, drug_1_idx, cell_line_idx) for
                                              drug_1_idx, drug_2_idx, cell_line_idx in
                                              temp_cross_validation_folds[fold]]
        # print(temp_cross_validation_folds[fold][0:10], cross_validation_folds_pos_drug_drug_edges[fold][0:10])
    cross_validation_folds_neg_drug_drug_edges = temp_cross_validation_folds

    non_drug_drug_edge_types = ['gene_gene', 'target_drug']
    cross_validation_folds_pos_non_drug_drug_edges = cross_val.create_cross_val_split_non_drug_drug_edges\
        (number_of_folds, adj_mats_init, non_drug_drug_edge_types)
    cross_validation_folds_neg_non_drug_drug_edges = cross_val.create_neg_cross_val_split_non_drug_drug_edges\
        (number_of_folds, adj_mats_init,non_drug_drug_edge_types)




    ######################## NODE FEATURE MATRIX CREATION ###########################################

    # featureless (genes)
    gene_feat = sp.identity(n_genes)
    gene_nonzero_feat, gene_num_feat = gene_feat.shape
    gene_feat = utils.sparse_to_tuple(gene_feat.tocoo())

    # features (drugs)
    use_drug_feat_options = synverse_settings['use_drug_feat']
    for use_drug_feat_option in use_drug_feat_options:
        drug_feat = prepare_drug_feat(drug_maccs_keys_feature_df, drug_node_2_idx, n_drugs, use_drug_feat_option)

        node_feat_dict = {'gene': gene_feat, 'drug' : drug_feat}
        ##########write training code here##################
        h_sizes = [64,32] #only hidden and output_layer


        for fold_no in range(number_of_folds):

            ###################################### Prepare DATA ########################################

            train_pos_edges_dict, train_neg_edges_dict, val_pos_edges_dict, val_neg_edges_dict = prepare_train_edges \
                (cross_validation_folds_pos_drug_drug_edges, cross_validation_folds_neg_drug_drug_edges, \
                 cross_validation_folds_pos_non_drug_drug_edges, cross_validation_folds_neg_non_drug_drug_edges,
                 fold_no, total_cell_lines)

            edge_type_wise_number_of_subtypes = {}
            for edge_type in edge_types:
                edge_type_wise_number_of_subtypes[edge_type] = len(train_pos_edges_dict[edge_type])

            encoder = Encoder(h_sizes, node_feat_dict, train_pos_edges_dict,edge_types)

            # init different decoder according to edge type
            # hardcode the decoder choice for now

            #change the drug_drug decoder to 'dedicom' later
            decoder_names = {'gene_gene': 'bilinear', 'target_drug': 'biliear', 'drug_target': 'bilinear', 'drug_drug':'dedicom'}

            decoders = {}
            for edge_type in edge_types:
                n_sub_types = edge_type_wise_number_of_subtypes[edge_type]
                if decoder_names=='bilinear':
                    decoders[edge_type] = BilinearDecoder(n_sub_types, h_sizes[-1])
                elif decoder_names=='dedicom':
                    decoders[edge_type] = DedicomDecoder(n_sub_types, h_sizes[-1])


            model = SynverseModel(encoder = encoder, decoders=decoders)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

            minibatch_handlder = MinibatchHandler(train_pos_edges_dict, batch_size, total_cell_lines)

            for epoch in range(epochs):
                #shuffle each training tensor  at the beginning of each epoch
                train_pos_edges_dict = minibatch_handlder.shuffle_train_edges(train_pos_edges_dict)
                train_neg_edges_dict = minibatch_handlder.shuffle_train_edges(train_neg_edges_dict)

                #split the train edges in chunks with size=batch_size
                #dict of list of split tensors/torches
                train_pos_edges_split_dict = {edge_type:[] for edge_type in edge_types}
                train_neg_edges_split_dict = {edge_type:[] for edge_type in edge_types}

                for edge_type in train_pos_edges_dict:
                    for i in range (len(train_pos_edges_dict[edge_type])):
                        train_pos_edges_split_dict[edge_type][i]= torch.split(train_pos_edges_dict[edge_type][i],batch_size,dim=1)
                        train_neg_edges_split_dict[edge_type][i] = torch.split(train_neg_edges_dict[edge_type][i],
                                                                               batch_size*neg_fact, dim=1)

                while not minibatch_handlder.is_batch_finished():
                    egde_type, edge_sub_type_idx, batch_num =  minibatch_handlder.next_minibatch()
                    if egde_type == 'drug_drug':
                        node_feat = drug_feat
                    elif egde_type == 'gene_gene':
                        node_feat = gene_feat
                    elif egde_type == 'target_drug':
                        node_feat = gene_feat
                    elif egde_type == 'drug_target':
                        node_feat = drug_feat

                    batch_pos_train_edges =  train_pos_edges_split_dict[edge_type][edge_sub_type_idx][batch_num]

                    batch_neg_train_edges = train_neg_edges_split_dict[edge_type][edge_sub_type_idx][batch_num]


                    training_batch_loss = train(model, optimizer, batch_pos_train_edges, batch_neg_train_edges, edge_type, edge_sub_type_idx)


                if epoch % 150 ==0:
                    for edge_type in edge_types:
                        for i in range(val_pos_edges_dict[edge_type]):
                            val_pos_edges=  val_pos_edges_dict[edge_type][i]
                            val_neg_edges = val_neg_edges_dict[edge_type][i]
                            loss = val(model, val_pos_edges, val_neg_edges, edge_type, i)
                            print(loss)







