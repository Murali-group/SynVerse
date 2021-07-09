from __future__ import division
from __future__ import print_function
from operator import itemgetter
from itertools import combinations
import time
import os
import random
import copy
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
from collections import defaultdict
import os.path as osp
import argparse
from tqdm import tqdm
import math
from itertools import combinations, permutations, product

import networkx as nx
import scipy.sparse as sp
from scipy.io import savemat, loadmat
from sklearn import metrics
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, precision_recall_curve

import torch
import torchvision
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, GAE, VGAE, SAGEConv, GMMConv, GATConv
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
from sklearn.decomposition import PCA
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.utils import to_undirected, negative_sampling
from torch_geometric.nn.models import InnerProductDecoder
from torch_geometric.nn.inits import reset

import models.synverse.utils as utils
import models.synverse.cross_validation as cross_val
from models.synverse.minibatch import MinibatchHandler
from models.synverse.BipartiteGCN import BipartiteGCN

import wandb
# wandb.login()

EPS = 1e-15
MAX_LOGVAR = 10
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def sigmoid(x):
    return 1. / (1 + np.exp(-x))


def compute_drug_drug_link_probability(cell_line_specific_edges_pos, cell_line_specific_edges_neg, cell_line,
                                       rec, val_fold, idx_2_drug_node):
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



def plot_loss(model, edge_type, pos_edges_split_dict,neg_edges_split_dict,\
              edge_type_wise_number_of_subtypes, idx_2_cell_line, train_or_val, min_loss, epoch, wandb_step):
    total_loss = 0
    for edge_sub_type in range(edge_type_wise_number_of_subtypes[edge_type]):
        cell_line_wise_loss = 0
        for split_idx in range(len(pos_edges_split_dict[edge_type][edge_sub_type])):
            pos_edges = pos_edges_split_dict[edge_type][edge_sub_type][split_idx].to(dev)
            neg_edges = neg_edges_split_dict[edge_type][edge_sub_type][split_idx].to(dev)
            batch_wise_pos_pred, batch_wise_neg_pred, loss = val(model, pos_edges,
                                                                       neg_edges, edge_type,
                                                                       edge_sub_type)
            cell_line_wise_loss += loss
        cell_line_wise_loss = cell_line_wise_loss/float(len(pos_edges_split_dict[edge_type][edge_sub_type]))
        total_loss += cell_line_wise_loss.to('cpu').detach().item()

        if edge_type == 'drug_drug':
            cell_line = idx_2_cell_line[edge_sub_type]
            cell_line_wise_title = train_or_val + '_loss_' + cell_line
            wandb.log({cell_line_wise_title: cell_line_wise_loss}, step=wandb_step)

    # if total_val_loss.to('cpu').detach().numpy()[0] < min_val_loss.to('cpu').detach().numpy()[0]:
    if total_loss < min_loss:
        # print(total_loss, min_loss)
        min_loss = total_loss

        # print(edge_type+': current minimum ' + train_or_val+ ' loss = ', min_loss, ' at epoch: ', epoch)

    wandb.log({train_or_val+ '_loss_'+ edge_type: total_loss}, step=wandb_step)

    return min_loss




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

        pos_loss = -torch.log(self.decoders[edge_type](z, batch_pos_edge_index,
                    edge_sub_type_idx, sigmoid=True) + EPS).mean()

        neg_loss = -torch.log(1 - self.decoders[edge_type](z, batch_neg_edge_index,
                    edge_sub_type_idx, sigmoid=True) + EPS).mean()

        # return pos_loss
        return pos_loss + neg_loss

    def predict(self,  z, batch_pos_edge_index, batch_neg_edge_index, edge_type, edge_sub_type_idx):
        #it will return two torch tensors with 4 rows. In each tensor,
        # first row =  source node
        # second row = target node
        # third row = predicted edge prob
        # forth row = true edge prob (0 or 1)
        pos_pred = self.decoders[edge_type](z, batch_pos_edge_index, edge_sub_type_idx, sigmoid=True)
        neg_pred = self.decoders[edge_type](z, batch_neg_edge_index, edge_sub_type_idx, sigmoid=True)

        pos_loss = -torch.log(pos_pred + EPS).mean()
        neg_loss = -torch.log(1 - neg_pred + EPS).mean()

        loss = pos_loss + neg_loss


        pos_y = torch.ones(batch_pos_edge_index.size(1)).to(dev)
        neg_y = torch.zeros(batch_neg_edge_index.size(1)).to(dev)

        pos_pred = torch.stack([batch_pos_edge_index[0], batch_pos_edge_index[1], pos_pred, pos_y], dim=0)
        neg_pred = torch.stack([batch_neg_edge_index[0], batch_neg_edge_index[1], neg_pred, neg_y], dim=0)


        return pos_pred, neg_pred, loss


    # def test(self, z, pos_edge_index, neg_edge_index):
    #     r"""Given latent variables :obj:`z`, positive edges
    #     :obj:`pos_edge_index` and negative edges :obj:`neg_edge_index`,
    #     computes area under the ROC curve (AUC) and average precision (AP)
    #     scores.
    #
    #     Args:
    #         z (Tensor): The latent space :math:`\mathbf{Z}`.
    #         pos_edge_index (LongTensor): The positive edges to evaluate
    #             against.
    #         neg_edge_index (LongTensor): The negative edges to evaluate
    #             against.
    #     """
    #     pos_y = z.new_ones(pos_edge_index.size(1))
    #     neg_y = z.new_zeros(neg_edge_index.size(1))
    #     y = torch.cat([pos_y, neg_y], dim=0)
    #
    #     pos_pred = self.decoder(z, pos_edge_index, sigmoid=True)
    #     neg_pred = self.decoder(z, neg_edge_index, sigmoid=True)
    #
    #     pred = torch.cat([pos_pred, neg_pred], dim=0)
    #
    #     y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()
    #     pr, rec, thresholds = precision_recall_curve(y, pred)
    #     pd.DataFrame([y, pred], index=['true', 'pred']).T.to_csv('preds.csv')
    #     pd.DataFrame([pr, rec, thresholds], index=['pr', 'rec', 'thres']).T.to_csv('pr.csv')
    #     return utils.precision_at_k(y, pred, pos_edge_index.size(1)), average_precision_score(y, pred)

class Encoder(torch.nn.Module): # in this function I have used one weight matrix for all drug_drug cell_lines
    #h_sizes is an array. containing the gradual node number decrease from initial hidden_layer output  to final output_dim.\
    # Input dim is not included
    def __init__(self, h_sizes, bias, dr, encoder_type, node_feat_dict, train_pos_edges_dict,  edge_types, edge_subtype_dict, n_drugs, n_genes):
        super(Encoder, self).__init__()

        self.edge_types = edge_types
        self.node_feat_dict = node_feat_dict
        self.train_pos_edges_dict = train_pos_edges_dict
        self.h_sizes = h_sizes
        self.num_hidden = len(h_sizes)
        self.hidden=nn.ModuleDict() #dict of dict. self.hidden[0]=> contains a dictionary for first hidden layer. In this dictionary, keys = edge_type,
        # value = GCN layer dedicated to it.
        for hid_layer_no in range(self.num_hidden):
            self.hidden[str(hid_layer_no)] = nn.ModuleDict()
            for edge_type in edge_types:
                if hid_layer_no == 0:
                    if edge_type in ['drug_drug', 'drug_target']:
                        input_feat_dim = list(node_feat_dict['drug'].size())[1]
                    else:
                        input_feat_dim = list(node_feat_dict['gene'].size())[1]

                    if encoder_type =='local':
                        self.hidden[str(hid_layer_no)][edge_type] =\
                            (EdgeTypeSpecGCNLayerLocal(input_feat_dim, h_sizes[hid_layer_no], \
                            bias, dr, edge_type, edge_subtype_dict[edge_type], n_drugs, n_genes))
                    elif encoder_type=='global':
                        self.hidden[str(hid_layer_no)][edge_type] = \
                            (EdgeTypeSpecGCNLayerGlobal(input_feat_dim, h_sizes[hid_layer_no], bias, dr, edge_type,
                                                        n_drugs, n_genes))
                else:
                    if encoder_type == 'local':
                        self.hidden[str(hid_layer_no)][edge_type] = \
                            (EdgeTypeSpecGCNLayerLocal(h_sizes[hid_layer_no - 1], h_sizes[hid_layer_no], bias, dr, edge_type, edge_subtype_dict[edge_type], n_drugs, n_genes))
                    elif encoder_type == 'global':
                        self.hidden[str(hid_layer_no)][edge_type] = \
                            (EdgeTypeSpecGCNLayerGlobal(h_sizes[hid_layer_no - 1], h_sizes[hid_layer_no], bias, dr,
                            edge_type, n_drugs, n_genes))


    def forward(self):
        # for one input layer
        # this will contain the output  i.e. embedding after hidden layer 1
        node_types = ['drug','gene']
        hidden_output = {}
        layer_input = copy.deepcopy(self.node_feat_dict)
        for hid_layer_no in range(self.num_hidden):
            hidden_output[hid_layer_no] = {node_type:[] for node_type in node_types}
            # hidden_output[1]={'drug': final_drug_embedding_from_hidden_layer_1, 'gene':final_gene_embedding_from_hidden_layer_1}
            # self.hidden1_output['drug'].append()
            for edge_type in self.edge_types:
                if edge_type in ['drug_drug', 'target_drug']:
                    #target_node = drug in both type of edges i.e.
                    # drug embedding is being computed. Hence, put under 'drug' key.
                    hidden_output[hid_layer_no]['drug'].append(
                        self.hidden[str(hid_layer_no)][edge_type](layer_input, self.train_pos_edges_dict))
                else:
                    hidden_output[hid_layer_no]['gene'].append(
                        self.hidden[str(hid_layer_no)][edge_type](layer_input, self.train_pos_edges_dict))

            for node_type in node_types:
                output = hidden_output[hid_layer_no][node_type][0]
                for i in range(1, len(hidden_output[hid_layer_no][node_type])):
                    output += hidden_output[hid_layer_no][node_type][i]
                hidden_output[hid_layer_no][node_type] = output #temporary commentout
                # hidden_output[hid_layer_no][node_type] = F.normalize(F.relu(output),dim=1, p=2)

            layer_input = hidden_output[hid_layer_no]

        return layer_input


class EdgeTypeSpecGCNLayerLocal(torch.nn.Module):
    def __init__(self, in_channel, out_channel, bias, dr, edge_type, n_edge_subtype, n_drugs, n_genes):
        '''
        edge_subtype(int) = number of subtypes in an edge type e.g. number of cell lines for drug_drug edges
        '''
        super(EdgeTypeSpecGCNLayerLocal, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.n_drugs = n_drugs
        self.n_genes = n_genes
        self.dr = dr
        self.n_edge_subtype = n_edge_subtype
        self.edge_type = edge_type
        self.gcn = nn.ModuleList()

        if edge_type in ['gene_gene', 'drug_drug']: #might change here to incorporate cellline spec weights in encoder
            for edge_subtype in range(n_edge_subtype):
                self.gcn.append(GCNConv(self.in_channel, self.out_channel, bias = bias, add_self_loops=True, cached=False))
        elif edge_type == 'target_drug':
            for edge_subtype in range(n_edge_subtype):
                self.gcn.append(BipartiteGCN(self.in_channel, self.out_channel, bias = bias, adj_shape = (n_genes, n_drugs)))
        elif (edge_type == 'drug_target'):
            for edge_subtype in range(n_edge_subtype):
                self.gcn.append(BipartiteGCN(self.in_channel, self.out_channel, bias = bias, adj_shape = (n_drugs, n_genes)))
        else:
            print('unknown edge type')

    def forward(self, node_feat_dict, train_pos_edges_dict):

        source_node = self.edge_type.split('_')[0].replace('target', 'gene')
        target_node = self.edge_type.split('_')[1].replace('target', 'gene')

        x = node_feat_dict[source_node]
        output = torch.zeros(list(node_feat_dict[target_node].size())[0], self.out_channel).to(dev)

        edge_index_list = train_pos_edges_dict[self.edge_type]  # might change here to incorporate cellline spec weights in encoder
        for subtype in range(len(edge_index_list)):
            x1 = F.dropout(x.float(), p=self.dr, training=True)
            x1 = F.relu(self.gcn[subtype](x1, edge_index_list[subtype])) #temporary commentout
            # x1 = self.gcn[subtype](x1, edge_index_list[subtype])
            output += x1
        # Nure: look into this normalization
        output = F.normalize(output, dim=1, p=2)  # p=2 means l2-normalization #temporary commentout

        # print('edge type done:', edge_type)
        return output

class EdgeTypeSpecGCNLayerGlobal(torch.nn.Module):  # in this function I have used one weight matrix for all drug_drug cell_lines
    def __init__(self, in_channel, out_channel, bias, dr, edge_type, n_drugs, n_genes):
        super(EdgeTypeSpecGCNLayerGlobal, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.n_drugs = n_drugs
        self.n_genes = n_genes
        self.dr = dr
        self.edge_type = edge_type

        if edge_type in ['gene_gene', 'drug_drug']: #might change here to incorporate cellline spec weights in encoder
            self.gcn = GCNConv(self.in_channel, self.out_channel, bias = bias, add_self_loops=True, cached=False)
        else:
            if(edge_type == 'target_drug'):
                self.gcn = BipartiteGCN(self.in_channel, self.out_channel, bias = bias, adj_shape = (n_genes, n_drugs))
            elif (edge_type == 'drug_target'):
                self.gcn = BipartiteGCN(self.in_channel, self.out_channel, bias = bias, adj_shape = (n_drugs, n_genes))
            else:
                print('unknown edge type')

    def forward(self, node_feat_dict, train_pos_edges_dict):

        source_node = self.edge_type.split('_')[0].replace('target', 'gene')
        target_node = self.edge_type.split('_')[1].replace('target', 'gene')


        x = node_feat_dict[source_node]
        output = torch.zeros(list(node_feat_dict[target_node].size())[0], self.out_channel).to(dev)


        edge_index_list = train_pos_edges_dict[self.edge_type] #might change here to incorporate cellline spec weights in encoder

        for edge_index in edge_index_list:
            x1 = F.dropout(x.float(), p=self.dr, training=True)
            x1 = F.relu(self.gcn(x1, edge_index))
            # x1 = F.dropout(x, p=0.2, training=True)
            # x1 = F.relu(self.gcn(x, adj_mat))
            output += x1
        #Nure: look into this normalization
        output = F.normalize(output, dim=1, p=2) #p=2 means l2-normalization

        # print('edge type done:', edge_type)
        return output





class BilinearDecoder(torch.nn.Module): #one decoder object for one edge_type
    def __init__(self, edge_type, n_sub_types, w_dim):
        # n_sub_types = how many different weight matrix is needed to initialize
        # w_dim = wight matrix dimension will be w_dim * w_dim i.e. dimension of the final nodel embedding
        super(BilinearDecoder, self).__init__()
        self.edge_type = edge_type
        self.w_dim = w_dim
        self.weight = nn.Parameter((utils.weight_matrix_glorot(w_dim, w_dim)).to(dev))

        # self.weights = nn.ParameterList()
        # for i in range(n_sub_types):
        #     self.weights.append(Parameter(utils.weight_matrix_glorot(w_dim, w_dim).to(dev)))



    def forward(self, z, batch_edges, edge_sub_type_idx, sigmoid=True):
        if self.edge_type == 'gene_gene':
            z1 = z['gene']
            z2 = z['gene']
        elif self.edge_type == 'target_drug':
            z1 = z['gene']
            z2 = z['drug']
        elif self.edge_type == 'drug_target':
            z1 = z['drug']
            z2 = z['gene']
        elif self.edge_type == 'drug_drug':
            z1 = z['drug']
            z2 = z['drug']


        row_embed = z1[batch_edges[0]]
        col_embed = z2[batch_edges[1]]

        # product_1 = torch.matmul(row_embed, self.weight)
        product_1 = torch.matmul(row_embed, self.weight)
        # print('in bilinear decoder: ',product_1.size(), col_embed.t().size())
        # print(self.weight[0])#, newWeight[850])
        # sys.exit()
        product_2 = torch.matmul(product_1, col_embed.t())
        score = torch.diagonal(product_2)

        return torch.sigmoid(score) if sigmoid else score

    def reset_parameters(self):
        self.weight.data.normal_(std=1 / np.sqrt(self.w_dim))


class DedicomDecoder(torch.nn.Module): #one decoder object for one edge_type
    def __init__(self, edge_type, n_sub_types, w_dim):
        '''
        params:
        1. edge_type= any of the one from the four edge types i.e. gene_gene, targte_drug, drug_target, drug_drug,
        2. n_sub_types = number of subtype e.g. drug_drug edge_type can consisted of multiple cell_line based sub_edge_type
        3. w_dim = dimension of the final node embedding.
                    wight matrix dimension  will be w_dim * w_dim

        function:
        1. initialize one global weight matrix for all subtypes of a particular edge_type i.e. for all drug_drug
        edges there will be one global_weight matrix
        2. initialize separate local weight matrix for each subtype of a particular edge_type. i.e.
        for drug_drug edges from each cell line  there will be one dedicated local_weight matrix.

        '''

        super(DedicomDecoder, self).__init__()
        self.edge_type = edge_type
        self.global_weight = nn.Parameter((utils.weight_matrix_glorot(w_dim, w_dim)).to(dev))
        self.local_weights = nn.ParameterList()

        for i in range(n_sub_types):
            diagonal_vals = torch.reshape(utils.weight_matrix_glorot(w_dim, 1), [-1])
            self.local_weights.append(Parameter(torch.diag(diagonal_vals).to(dev)))


    def forward(self, z, batch_edges, edge_sub_type_idx, sigmoid=True):
        if self.edge_type == 'gene_gene':
            z1 = z['gene']
            z2 = z['gene']
        elif self.edge_type == 'target_drug':
            z1 = z['gene']
            z2 = z['drug']
        elif self.edge_type == 'drug_target':
            z1 = z['drug']
            z2 = z['gene']
        elif self.edge_type == 'drug_drug':
            z1 = z['drug']
            z2 = z['drug']
        row_embed = z1[batch_edges[0]]
        col_embed = z2[batch_edges[1]]

        # print(row_embed.size(), self.local_weights[edge_sub_type_idx].size())
        product_1 = torch.matmul(row_embed, self.local_weights[edge_sub_type_idx])
        product_2 = torch.matmul(product_1, self.global_weight)
        product_3 = torch.matmul(product_2, self.local_weights[edge_sub_type_idx])
        product_4 = torch.matmul(product_3, col_embed.t())
        score = torch.diagonal(product_4)
        return torch.sigmoid(score) if sigmoid else score

    # def reset_parameters(self):
    #     self.weight.data.normal_(std=1 / np.sqrt(self.in_dim))



class FeedforwardNeuralNetModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeedforwardNeuralNetModel, self).__init__()
        # Linear function
        self.fc1 = nn.Linear(input_dim, hidden_dim)

        # Non-linearity
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        # Linear function (readout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Linear function  # LINEAR
        out = self.fc1(x)

        # Non-linearity  # NON-LINEAR
        out = self.relu(out)

        # Linear function (readout)  # LINEAR
        out = self.fc2(out)
        return out


class NNDecoder(torch.nn.Module):
    #this decoder is applicable for only drug_drug edge prediction
    def __init__(self, edge_type, n_sub_types, w_dim,gene_expression_df):
        super(NNDecoder, self).__init__()
        self.edge_type = edge_type
        self.gene_expression_df = gene_expression_df

        self.hidden_layer_1 = 512
        self.out_layer = 1
        # self.feed_forward_model = FeedforwardNeuralNetModel\
        #     (concatenated_drug_drug_genex_embedding.size()[1], hidden_layer_1, out_layer)
        self.feed_forward_model = FeedforwardNeuralNetModel \
            (w_dim, self.hidden_layer_1, self.out_layer).to(dev)

    def forward(self, z, batch_edges, edge_sub_type_idx, sigmoid=True):

        if self.edge_type == 'drug_drug':
            z1 = z['drug']
            z2 = z['drug']
            drug1_embed = z1[batch_edges[0]].to(dev)
            drug2_embed = z2[batch_edges[1]].to(dev)
            gene_expression = torch.tensor\
                ([self.gene_expression_df.iloc[edge_sub_type_idx]]*(drug1_embed.size()[0])).to(dev)
            concatenated_drug_drug_genex_embedding =\
                torch.cat((drug1_embed, drug2_embed, gene_expression), axis=1).\
                type(torch.FloatTensor).to(dev)

            # print('size: ', concatenated_drug_drug_genex_embedding.size())

            #now feed the input to nn model
            score = self.feed_forward_model(concatenated_drug_drug_genex_embedding)
            score = torch.flatten(score)
            return torch.sigmoid(score) if sigmoid else score


# class TFDecoder(torch.nn.Module):
#     def __init__(self, num_nodes, TFIDs):
#         super(TFDecoder, self).__init__()
#         self.TFIDs = list(TFIDs)
#         self.num_nodes = num_nodes
#         self.in_dim = 1  # one relation type
#         # self.weight = nn.Parameter(torch.Tensor(len(self.TFIDs)))
#         self.weight = nn.Parameter(torch.Tensor(self.num_nodes))
#         self.reset_parameters()
#val
#     def forward(self, z, edge_index, sigmoid=True):
#         # newWeight = torch.zeros(self.num_nodes).to(dev)
#         # nCnt = 0
#         # for idx in self.TFIDs:
#         #    newWeight[idx] = self.weight[nCnt]
#         #    nCnt += 1
#         zNew = torch.mul(z.t(), self.weight).t()
#         # print(self.weight[0])#, newWeight[850])
#         # sys.exit()
#         value = (zNew[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
#
#         return torch.sigmoid(value) if sigmoid else value
#
#     def reset_parameters(self):
#         self.weight.data.normal_(std=1 / np.sqrt(self.in_dim))
#
#
# class RESCALDecoder(torch.nn.Module):
#     def __init__(self, out_dim):
#         super(RESCALDecoder, self).__init__()
#         self.out_dim = out_dim
#         self.in_dim = 1  # one relation type
#         self.weight = nn.Parameter(torch.Tensor(self.out_dim, self.out_dim))
#         self.reset_parameters()
#
#     def forward(self, z, edge_index, sigmoid=True):
#         # zNew = z.clone()*self.weight
#         zNew = torch.matmul(z.clone(), self.weight)
#         # zNew = z*self.weight
#         # print(edge_index)
#         # print(zNew.shape,self.weight)
#         value = (zNew[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
#
#         # self.weight[edge_index[0]]
#         # print(value, edge_index)
#         # value = (1 * z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
#         # print(edge_index.shape,value.shape)
#         return torch.sigmoid(value) if sigmoid else value
#
#     def reset_parameters(self):
#         self.weight.data.normal_(std=1 / np.sqrt(self.in_dim))


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
    pos_pred, neg_pred, val_loss = model.predict(z, pos_edge_index, neg_edge_index, edge_type, edge_sub_type_idx)
    return pos_pred, neg_pred, val_loss


def prepare_drug_feat(drug_maccs_keys_feature_df, drug_node_2_idx, n_drugs, use_drug_feat_option):
    if use_drug_feat_option:
        drug_maccs_keys_feature_df['drug_idx'] = drug_maccs_keys_feature_df['pubchem_cid']. \
            apply(lambda x: drug_node_2_idx[x])
        drug_maccs_keys_feature_df = drug_maccs_keys_feature_df.sort_values(by=['drug_idx'])
        assert len(drug_maccs_keys_feature_df) == n_drugs, 'problem in drug feat creation'
        drug_feat = torch.tensor(drug_maccs_keys_feature_df.drop(columns=['pubchem_cid']).set_index('drug_idx').to_numpy())
        # drug_num_feat = drug_feat.shape[1]
        # drug_nonzero_feat = np.count_nonzero(drug_feat)
        #
        # drug_feat = sp.csr_matrix(drug_feat)
        # drug_feat = utils.sparse_to_tuple(drug_feat.tocoo())

    else:
        # #one hot encoding for drug features
        drug_feat = torch.tensor(np.identity(n_drugs))

        # drug_nonzero_feat, drug_num_feat = drug_feat.shape
        # drug_feat = utils.sparse_to_tuple(drug_feat.tocoo())

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
        train_pos_edges = torch.stack([torch.LongTensor(train_pos_source_nodes),torch.LongTensor(train_pos_target_nodes)],dim=0).to(dev)

        train_neg_source_nodes = [d1 for d1, d2, c in train_neg_drug_drug_edges_set if c == cell_line_idx]
        train_neg_target_nodes = [d2 for d1, d2, c in train_neg_drug_drug_edges_set if c == cell_line_idx]
        train_neg_edges = torch.stack([torch.LongTensor(train_neg_source_nodes), torch.LongTensor(train_neg_target_nodes)], dim=0).to(dev)

        val_pos_source_nodes = [d1 for d1, d2, c in val_pos_drug_drug_edges_set if c == cell_line_idx]
        val_pos_target_nodes = [d2 for d1, d2, c in val_pos_drug_drug_edges_set if c == cell_line_idx]
        val_pos_edges = torch.stack(
            [torch.LongTensor(val_pos_source_nodes), torch.LongTensor(val_pos_target_nodes)], dim=0).to(dev)

        val_neg_source_nodes = [d1 for d1, d2, c in val_neg_drug_drug_edges_set if c == cell_line_idx]
        val_neg_target_nodes = [d2 for d1, d2, c in val_neg_drug_drug_edges_set if c == cell_line_idx]
        val_neg_edges = torch.stack(
            [torch.LongTensor(val_neg_source_nodes), torch.LongTensor(val_neg_target_nodes)], dim=0).to(dev)

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
            [torch.LongTensor(train_pos_source_nodes), torch.LongTensor(train_pos_target_nodes)], dim=0).to(dev)

        train_neg_source_nodes = [d1 for d1, d2 in train_neg_non_drug_drug_edges_set]
        train_neg_target_nodes = [d2 for d1, d2 in train_neg_non_drug_drug_edges_set]
        train_neg_edges = torch.stack(
            [torch.LongTensor(train_neg_source_nodes), torch.LongTensor(train_neg_target_nodes)], dim=0).to(dev)

        val_pos_source_nodes = [d1 for d1, d2 in val_pos_non_drug_drug_edges_set]
        val_pos_target_nodes = [d2 for d1, d2 in val_pos_non_drug_drug_edges_set]
        val_pos_edges = torch.stack(
            [torch.LongTensor(val_pos_source_nodes), torch.LongTensor(val_pos_target_nodes)], dim=0).to(dev)

        val_neg_source_nodes = [d1 for d1, d2 in val_neg_non_drug_drug_edges_set]
        val_neg_target_nodes = [d2 for d1, d2  in val_neg_non_drug_drug_edges_set]
        val_neg_edges = torch.stack(
            [torch.LongTensor(val_neg_source_nodes), torch.LongTensor(val_neg_target_nodes)], dim=0).to(dev)

        train_pos_edges_dict[edge_type].append(train_pos_edges)
        train_neg_edges_dict[edge_type].append(train_neg_edges)
        val_pos_edges_dict[edge_type].append(val_pos_edges)
        val_neg_edges_dict[edge_type].append(val_neg_edges)

    return train_pos_edges_dict, train_neg_edges_dict, val_pos_edges_dict, val_neg_edges_dict



def prepare_pred_score_for_saving(pos_pred, neg_pred, cell_line, idx_2_drug_node):
    pos_edge_dict = {'drug_1_idx': [], 'drug_2_idx': [], 'cell_line': [], 'predicted': [], 'true': []}
    neg_edge_dict = {'drug_1_idx': [], 'drug_2_idx': [], 'cell_line': [], 'predicted': [], 'true': []}


    pos_edge_dict['drug_1_idx'] = pos_pred[0].detach().cpu().numpy()
    pos_edge_dict['drug_2_idx'] = pos_pred[1].detach().cpu().numpy()
    pos_edge_dict['cell_line'] = np.array([cell_line]*pos_pred.size()[1])
    pos_edge_dict['predicted'] =  pos_pred[2].detach().cpu().numpy()
    pos_edge_dict['true'] =  pos_pred[3].detach().cpu().numpy()

    neg_edge_dict['drug_1_idx'] =  neg_pred[0].detach().cpu().numpy()
    neg_edge_dict['drug_2_idx'] = neg_pred[1].detach().cpu().numpy()
    neg_edge_dict['cell_line'] = np.array([cell_line]*neg_pred.size()[1])
    neg_edge_dict['predicted'] =  neg_pred[2].detach().cpu().numpy()
    neg_edge_dict['true'] = neg_pred[3].detach().cpu().numpy()

    pos_df = pd.DataFrame.from_dict(pos_edge_dict)
    neg_df = pd.DataFrame.from_dict(neg_edge_dict)

    pos_df['drug_1'] = pos_df['drug_1_idx'].apply(lambda x: idx_2_drug_node[x])
    pos_df['drug_2'] = pos_df['drug_2_idx'].apply(lambda x: idx_2_drug_node[x])

    neg_df['drug_1'] = neg_df['drug_1_idx'].apply(lambda x: idx_2_drug_node[x])
    neg_df['drug_2'] = neg_df['drug_2_idx'].apply(lambda x: idx_2_drug_node[x])

    pos_df = pos_df[['drug_1', 'drug_2', 'cell_line', 'predicted', 'true']].sort_values(by=['predicted'], \
                                                                                        ascending=False)
    neg_df = neg_df[['drug_1', 'drug_2', 'cell_line', 'predicted', 'true']].sort_values(by=['predicted'], \
                                                                                        ascending=False)
    return pos_df, neg_df

def save_drug_drug_link_probability(pos_df, neg_df, encoder_type, decoder_type, h_sizes, use_drug_feat_option, lr, \
                                    epochs, batch_size, dr, out_dir, drug_based_batch_end):
    #inputs: df with link prediction probability after applying sigmoid on models predicted score for an edges(positive, negative)

    if drug_based_batch_end:
        extra_direction_on_out_dir = 'drug_based_batch_end/'
    else:
        extra_direction_on_out_dir = ''

    pos_out_file = out_dir+\
                    extra_direction_on_out_dir + 'pos_val_scores'+'_encoder_' +encoder_type+'_decoder_' + decoder_type + '_hsize_'+str(h_sizes)+\
                   '_drugfeat_'+ str(use_drug_feat_option)+'_e_'+str(epochs) +'_lr_'+str(lr) +'_batch_'+ str(batch_size)\
                   +'_dr_'+ str(dr)+'.tsv'
    neg_out_file = out_dir + extra_direction_on_out_dir + 'neg_val_scores'+ '_encoder_' +encoder_type+'_decoder_' + decoder_type + '_hsize_'+str(h_sizes)+\
                   '_drugfeat_'+str(use_drug_feat_option)+'_e_'+str(epochs) +'_lr_'+str(lr) +'_batch_'+ str(batch_size) +\
                   '_dr_'+ str(dr)+'.tsv'

    os.makedirs(os.path.dirname(pos_out_file), exist_ok=True)
    os.makedirs(os.path.dirname(neg_out_file), exist_ok=True)

    pos_df.to_csv(pos_out_file, sep='\t')
    neg_df.to_csv(neg_out_file, sep='\t')

def train_log(loss, wandb_step, edge_type, edge_name):
    loss = float(loss)
    edge_type_idx = utils.edge_type_to_idx(edge_type)
    l = 'batch_wise_loss_'+ edge_name
    wandb.log({l: loss, 'edge_type':edge_type_idx },\
              step = wandb_step)
    # wandb.log({'Epoch': epoch, 'loss': loss}, step=wandb_step)



def run_synverse_model(ppi_sparse_matrix, gene_node_2_idx, drug_target_df, drug_maccs_keys_feature_df, \
                       gene_expression_feature_df, synergy_df, non_synergy_df,\
                       cell_line_2_idx, idx_2_cell_line,
                       cross_validation_folds_pos_drug_drug_edges, cross_validation_folds_neg_drug_drug_edges,\
                       cross_val_dir, neg_sampling_type, encoder_type, dd_decoder_type, out_dir, config_map, use_drug_based_batch_end):

    t1 = time.time()
    #model setup
    synverse_settings = config_map['ml_models_settings']['algs']['synverse']
    h_sizes = synverse_settings['h_sizes'] # only hidden and output_layer
    lr = synverse_settings['learning_rate']
    epochs = synverse_settings['epochs']
    # hidden1 = 64
    # hidden2 = 32
    # weight_decay = synverse_settings['weight_decay']
    dr = synverse_settings['dropout']
    # max_margin = synverse_settings['max_margin']
    batch_size = synverse_settings['batch_size']
    bias = synverse_settings['bias']

    neg_fact = config_map['ml_models_settings']['cross_val']['neg_fact']

    number_of_folds = config_map['ml_models_settings']['cross_val']['folds']



    gene_adj = nx.adjacency_matrix(nx.convert_matrix.from_scipy_sparse_matrix(ppi_sparse_matrix,\
                                                create_using=nx.Graph(),edge_attribute=None))
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
    # cell_line_2_idx = {cell_line: i for i, cell_line in enumerate(cell_lines)}
    # idx_2_cell_line = {i: cell_line for i, cell_line in enumerate(cell_lines)}
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

    # drug_degrees_list = [np.array(drug_adj.sum(axis=0)).squeeze() for drug_adj in drug_drug_adj_list]
    #### in  each drug_drug_adjacency_matrix in drug_degrees_list, if (x,y) is present then (y,x) is also present there.
    # drug_degrees_list = [np.array(drug_adj.sum(axis=0)).squeeze() for drug_adj in drug_drug_adj_list]

    print('finished network building')

    print('In Final network:\n Genes:%d Targetted Genes:%d  Drugs:%d' % (n_genes, tagetted_genes, n_drugs))

    # t2 = time.time()
    # print('time for network building: ', t2-t1)

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
    neg_edges_all_cell_line = list(zip(non_synergy_df['Drug1_idx'], non_synergy_df['Drug2_idx'], non_synergy_df['Cell_line_idx']))
    for fold in cross_validation_folds_neg_drug_drug_edges:
        # print('test: ', fold, len(cross_validation_folds[fold]), cross_validation_folds[fold])
        temp_cross_validation_folds[fold] = [neg_edges_all_cell_line[x] for x in
                                             cross_validation_folds_neg_drug_drug_edges[fold]]
        temp_cross_validation_folds[fold] += [(drug_2_idx, drug_1_idx, cell_line_idx) for
                                              drug_1_idx, drug_2_idx, cell_line_idx in
                                              temp_cross_validation_folds[fold]]
        # print(temp_cross_validation_folds[fold][0:10], cross_validation_folds_pos_drug_drug_edges[fold][0:10])
    cross_validation_folds_neg_drug_drug_edges = temp_cross_validation_folds

    non_drug_drug_edge_types = ['gene_gene', 'drug_target']
    # non_drug_drug_edge_types = ['drug_target']
    cross_validation_folds_pos_non_drug_drug_edges = cross_val.create_cross_val_split_non_drug_drug_edges\
        (number_of_folds, adj_mats_init, non_drug_drug_edge_types, cross_val_dir)
    # cross_validation_folds_neg_non_drug_drug_edges = cross_val.create_degree_dist_neg_cross_val_split_non_drug_drug_edges\
    #     (number_of_folds, adj_mats_init, non_drug_drug_edge_types, neg_fact, cross_val_dir)
    if neg_sampling_type == 'degree_based':
        cross_validation_folds_neg_non_drug_drug_edges = cross_val.create_degree_based_neg_cross_val_split_non_drug_drug_edges\
            (number_of_folds, adj_mats_init, non_drug_drug_edge_types, neg_fact, cross_val_dir)
    elif neg_sampling_type == 'semi_random':
        cross_validation_folds_neg_non_drug_drug_edges = cross_val.create_semi_random_neg_cross_val_split_non_drug_drug_edges\
            (number_of_folds, adj_mats_init, non_drug_drug_edge_types, neg_fact, cross_val_dir)


    ######################## NODE FEATURE MATRIX CREATION ###########################################

    # featureless (genes)
    gene_feat = torch.tensor(np.identity(n_genes)).to(dev)

    # features (drugs)
    use_drug_feat_options = synverse_settings['use_drug_feat']

    model_no = 0
    for use_drug_feat_option in use_drug_feat_options:
        drug_feat = prepare_drug_feat(drug_maccs_keys_feature_df, drug_node_2_idx, n_drugs, use_drug_feat_option)
        drug_feat = drug_feat.to(dev)

        # t4 = time.time()
        # print('time for drug feat preparation: ', t4 - t3)

        node_feat_dict = {'gene': gene_feat, 'drug' : drug_feat}
        ##########write training code here##################

        pos_df = pd.DataFrame()
        neg_df = pd.DataFrame()
        for fold_no in range(number_of_folds):
            with wandb.init(project='synverse_gpu3_genex_1', config=config_map):
                config_map = wandb.config
                ###################################### Prepare DATA ########################################

                train_pos_edges_dict, train_neg_edges_dict, val_pos_edges_dict, val_neg_edges_dict = prepare_train_edges\
                    (cross_validation_folds_pos_drug_drug_edges, cross_validation_folds_neg_drug_drug_edges,
                     cross_validation_folds_pos_non_drug_drug_edges, cross_validation_folds_neg_non_drug_drug_edges,
                     fold_no, total_cell_lines)

                # validation
                val_pos_edges_split_dict = {edge_type: [] for edge_type in edge_types}
                val_neg_edges_split_dict = {edge_type: [] for edge_type in edge_types}

                for edge_type in val_pos_edges_dict:
                    for i in range(len(val_pos_edges_dict[edge_type])):
                        val_pos_edges_split_dict[edge_type].append(torch.split(val_pos_edges_dict[edge_type][i],
                                        batch_size, dim=1))
                        val_neg_edges_split_dict[edge_type].append(torch.split(val_neg_edges_dict[edge_type][i],
                                        batch_size * neg_fact, dim=1))

                # t5 = time.time()
                # print('time for training matrix preparation: ', t5 - t4)

                # print('before split 1', train_pos_edges_dict['gene_gene'][0].size())
                edge_type_wise_number_of_subtypes = {}
                for edge_type in edge_types:
                    edge_type_wise_number_of_subtypes[edge_type] = len(train_pos_edges_dict[edge_type])



                encoder = Encoder(h_sizes, bias , dr, encoder_type,node_feat_dict, train_pos_edges_dict,edge_types,edge_type_wise_number_of_subtypes,\
                                  n_drugs, n_genes)

                decoder_names = {'gene_gene': 'bilinear', 'target_drug': 'bilinear',\
                                 'drug_target': 'bilinear', 'drug_drug':dd_decoder_type}

                decoders = nn.ModuleDict()
                for edge_type in edge_types:
                    n_sub_types = edge_type_wise_number_of_subtypes[edge_type]

                    if decoder_names[edge_type]=='bilinear':
                        decoders[edge_type] = BilinearDecoder(edge_type, n_sub_types, h_sizes[-1])

                    elif decoder_names[edge_type]=='dedicom':
                        decoders[edge_type] = DedicomDecoder(edge_type, n_sub_types, h_sizes[-1])

                    elif decoder_names[edge_type] == 'nndecoder':
                        #the size of the input is concatenation of (drug1, drug2 , cell_line_spec_gene_expression)
                        #here, h_sizes[-1] is the output dimension of embedding layer i.e. final dim of embedded drugs
                        # and len(gene_expression_feature_df.columns) is the #of gene expression features
                        decoders[edge_type] = NNDecoder(edge_type,n_sub_types,h_sizes[-1]*2 + \
                                            len(gene_expression_feature_df.columns), gene_expression_feature_df)


                model = SynverseModel(encoder = encoder, decoders=decoders).to(dev)
                print('model_no: ',model_no)
                print('fold_no: ',fold_no)
                model_no +=1

                model_param = model.state_dict()
                # model_param = model.parameters()
                # print(model_param)
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)

                print('learning rate: ', lr, 'h_sizes: ', h_sizes, 'optimizer: ', optimizer)

                minibatch_handlder = MinibatchHandler(train_pos_edges_dict, batch_size, total_cell_lines)

                wandb.watch(model, log='all', log_freq=1)

                wandb_step = 0

                min_loss_dd_train = 1000
                min_loss_dd_val = 1000
                min_loss_gg_train = 1000
                min_loss_gg_val = 1000

                min_loss_dg_train = 1000
                min_loss_dg_val = 1000
                min_loss_gd_train = 1000
                min_loss_gd_val = 1000

                for epoch in range(1, epochs+1):
                    t10 = time.time()
                    #shuffle each training tensor  at the beginning of each epoch
                    train_pos_edges_dict = minibatch_handlder.shuffle_train_edges(train_pos_edges_dict)
                    train_neg_edges_dict = minibatch_handlder.shuffle_train_edges(train_neg_edges_dict)


                    #split the train edges in chunks with size=batch_size
                    #dict of list of split tensors/torches
                    train_pos_edges_split_dict = {edge_type:[] for edge_type in edge_types}
                    train_neg_edges_split_dict = {edge_type:[] for edge_type in edge_types}

                    for edge_type in train_pos_edges_dict:
                        for i in range(len(train_pos_edges_dict[edge_type])):
                            train_pos_edges_split_dict[edge_type].append(torch.split(train_pos_edges_dict[edge_type][i],
                                                                                   batch_size, dim=1))
                            train_neg_edges_split_dict[edge_type].append(torch.split(train_neg_edges_dict[edge_type][i],
                                                                                   batch_size * neg_fact, dim=1))


                    batch_count = 0


                    # while not minibatch_handlder.is_batch_finished():
                    while True:
                        if use_drug_based_batch_end:
                            if minibatch_handlder.is_batch_finished_new():
                                break
                        else:
                            if minibatch_handlder.is_batch_finished():
                                break

                        # t13 = time.time()
                        e, edge_sub_type_idx, batch_num = minibatch_handlder.next_minibatch()
                        # t14 = time.time()
                        # print('time for next minibatch choosing: ', t14-t13)
                        # print('egde_type:', e, 'edge_sub_type_idx:', edge_sub_type_idx, 'batch_num:', batch_num)
                        # print('before split', train_pos_edges_dict['gene_gene'][0].size())
                        # print('before split', train_pos_edges_dict[e][edge_sub_type_idx].size())
                        #
                        # print('train_pos_edges_split_dict:', type(train_pos_edges_split_dict[e][edge_sub_type_idx]),len(train_pos_edges_split_dict[e][edge_sub_type_idx]))
                        # print('train_neg_edges_split_dict:', type(train_neg_edges_split_dict[e][edge_sub_type_idx]),len(train_neg_edges_split_dict[e][edge_sub_type_idx]))

                        batch_pos_train_edges = train_pos_edges_split_dict[e][edge_sub_type_idx][batch_num].to(dev)

                        batch_neg_train_edges = train_neg_edges_split_dict[e][edge_sub_type_idx][batch_num].to(dev)

                        # print(e, 'batches shapes: ', batch_pos_train_edges.shape, batch_neg_train_edges.shape )
                        training_batch_loss = train(model, optimizer, batch_pos_train_edges, batch_neg_train_edges,\
                                                    e, edge_sub_type_idx)

                        # t15 = time.time()
                        # print('time for batchwise training ', t15 - t14)
                        wandb_step+=1
                        # if e == 'drug_drug':
                        #
                        #     cell_line = idx_2_cell_line[edge_sub_type_idx]
                        #     train_log(training_batch_loss, wandb_step, e, cell_line)
                        # if e == 'gene_gene':
                        #     # wandb_step += 1
                        #     train_log(training_batch_loss, wandb_step, e, 'gene_gene')

                        wandb.log({'Epoch': epoch}, step=wandb_step)


                        # batch_count += 1

                    print('epoch: ', epoch, ' epoch time: ', time.time()-t10)

                    ##train and val loss plot after whole epoch
                    if epoch % 2 == 0:
                        min_loss_dd_train = plot_loss(model, 'drug_drug', train_pos_edges_split_dict, train_neg_edges_split_dict, \
                                  edge_type_wise_number_of_subtypes, idx_2_cell_line, 'train', min_loss_dd_train, epoch, wandb_step)
                        min_loss_gg_train = plot_loss(model, 'gene_gene', train_pos_edges_split_dict, train_neg_edges_split_dict, \
                                  edge_type_wise_number_of_subtypes, idx_2_cell_line, 'train', min_loss_gg_train, epoch, wandb_step)

                        min_loss_dg_train = plot_loss(model, 'drug_target', train_pos_edges_split_dict,
                                                      train_neg_edges_split_dict,
                                                      edge_type_wise_number_of_subtypes, idx_2_cell_line, 'train',
                                                      min_loss_dg_train, epoch, wandb_step)
                        min_loss_gd_train = plot_loss(model, 'target_drug', train_pos_edges_split_dict,
                                                      train_neg_edges_split_dict,
                                                      edge_type_wise_number_of_subtypes, idx_2_cell_line, 'train',
                                                      min_loss_gd_train, epoch, wandb_step)

                        min_loss_dd_val = plot_loss(model, 'drug_drug', val_pos_edges_split_dict, val_neg_edges_split_dict, \
                                  edge_type_wise_number_of_subtypes, idx_2_cell_line, 'val', min_loss_dd_val, epoch, wandb_step)
                        min_loss_gg_val = plot_loss(model, 'gene_gene', val_pos_edges_split_dict, val_neg_edges_split_dict, \
                                  edge_type_wise_number_of_subtypes, idx_2_cell_line, 'val', min_loss_gg_val, epoch, wandb_step)
                        min_loss_dg_val = plot_loss(model, 'drug_target', val_pos_edges_split_dict,
                                                    val_neg_edges_split_dict,
                                                    edge_type_wise_number_of_subtypes, idx_2_cell_line, 'val',
                                                    min_loss_dg_val, epoch, wandb_step)
                        min_loss_gd_val = plot_loss(model, 'target_drug', val_pos_edges_split_dict,
                                                    val_neg_edges_split_dict,
                                                    edge_type_wise_number_of_subtypes, idx_2_cell_line, 'val',
                                                    min_loss_gd_val, epoch, wandb_step)


                val_edge_type = 'drug_drug'
                total_val_loss = 0
                for edge_sub_type in range(edge_type_wise_number_of_subtypes[val_edge_type]):
                    cell_line = idx_2_cell_line[edge_sub_type]
                    for split_idx in range(len(val_pos_edges_split_dict[val_edge_type][edge_sub_type])):
                        val_pos_edges = val_pos_edges_split_dict[val_edge_type][edge_sub_type][split_idx].to(dev)
                        val_neg_edges = val_neg_edges_split_dict[val_edge_type][edge_sub_type][split_idx].to(dev)
                        batch_wise_pos_pred, batch_wise_neg_pred, val_loss = val(model, val_pos_edges, val_neg_edges,\
                                                                                 val_edge_type, edge_sub_type)
                        total_val_loss += val_loss

                        batch_wise_pos_df, batch_wise_neg_df = prepare_pred_score_for_saving\
                            (batch_wise_pos_pred, batch_wise_neg_pred, cell_line, idx_2_drug_node)
                        pos_df = pd.concat([pos_df, batch_wise_pos_df], axis=0)
                        neg_df = pd.concat([neg_df, batch_wise_neg_df], axis=0)
                    print('cell_line: ', cell_line )
                    # print('\n')

            save_drug_drug_link_probability(pos_df, neg_df, encoder_type, dd_decoder_type, h_sizes, use_drug_feat_option,\
                                            lr, epochs, batch_size, dr, out_dir, use_drug_based_batch_end)




