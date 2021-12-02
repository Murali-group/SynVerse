import copy
import scipy.sparse as sp
import numpy as np
from typing import Optional, Tuple

from torch_geometric.typing import Adj, OptTensor, PairTensor
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
from torch_sparse import SparseTensor, matmul, fill_diag, sum, mul
from torch_geometric.nn import GCNConv, RGCNConv, GATConv,SAGEConv
from torch_geometric.nn.conv import MessagePassing

import models.gnn.gnn_utils as utils

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

conv_dict = {'GCN': GCNConv, 'SAGE': SAGEConv, 'GAT': GATConv}


class Encoder(torch.nn.Module): # in this function I have used one weight matrix for all drug_drug cell_lines
    #h_sizes is an array. containing the gradual node number decrease from initial hidden_layer output  to final output_dim.\
    # Input dim is not included
    def __init__(self, h_sizes, bias, dr, encoder_type, conv_type, node_feat_dict, train_pos_edges_dict,  edge_types, edge_subtype_dict, n_drugs, n_genes):
        super(Encoder, self).__init__()

        self.edge_types = edge_types
        self.node_feat_dict = node_feat_dict
        self.train_pos_edges_dict = train_pos_edges_dict
        self.h_sizes = h_sizes
        self.num_hidden = len(h_sizes)
        self.encoder_type = encoder_type
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

                    if encoder_type == 'local':
                        self.hidden[str(hid_layer_no)][edge_type] =\
                            (ConvLayerLocal(conv_type, input_feat_dim, h_sizes[hid_layer_no], \
                                            bias, dr, edge_type, edge_subtype_dict[edge_type], n_drugs, n_genes))
                    elif encoder_type == 'local_v2':
                        self.hidden[str(hid_layer_no)][edge_type] =\
                            (ConvLayerLocal_v2(conv_type, input_feat_dim, h_sizes[hid_layer_no], \
                                            bias, dr, edge_type, edge_subtype_dict[edge_type], n_drugs, n_genes))
                    elif encoder_type == 'global':
                        self.hidden[str(hid_layer_no)][edge_type] = \
                            (ConvLayerGlobal(conv_type, input_feat_dim, h_sizes[hid_layer_no], bias, dr, edge_type,
                                             n_drugs, n_genes))

                else:
                    if encoder_type == 'local':
                        self.hidden[str(hid_layer_no)][edge_type] = \
                            (ConvLayerLocal(conv_type, h_sizes[hid_layer_no - 1], h_sizes[hid_layer_no], bias, dr,\
                            edge_type, edge_subtype_dict[edge_type], n_drugs, n_genes))
                    elif encoder_type == 'local_v2':
                        self.hidden[str(hid_layer_no)][edge_type] = \
                            (ConvLayerLocal_v2(conv_type, h_sizes[hid_layer_no - 1], h_sizes[hid_layer_no], bias, dr,
                                            edge_type, edge_subtype_dict[edge_type], n_drugs, n_genes))
                    elif encoder_type == 'global':
                        self.hidden[str(hid_layer_no)][edge_type] = \
                            (ConvLayerGlobal(conv_type, h_sizes[hid_layer_no - 1], h_sizes[hid_layer_no], bias, dr,
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
                if self.encoder_type == 'local':
                    hidden_output[hid_layer_no][node_type] = output #use this line when ConvLayerLocal is used
                elif self.encoder_type == 'local_v2':
                    hidden_output[hid_layer_no][node_type] = F.normalize(F.relu(output),dim=1, p=2)

            layer_input = hidden_output[hid_layer_no]

        return layer_input


class ConvLayerLocal(torch.nn.Module):
    def __init__(self,conv_type,  in_channel, out_channel, bias, dr, edge_type, n_edge_subtype, n_drugs, n_genes):
        '''
        edge_subtype(int) = number of subtypes in an edge type e.g. number of cell lines for drug_drug edges
        '''
        super(ConvLayerLocal, self).__init__()
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
                self.gcn.append(conv_dict[conv_type](self.in_channel, self.out_channel, bias = bias))
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

        edge_index_list = train_pos_edges_dict[self.edge_type]
        for subtype in range(len(edge_index_list)):
            x1 = F.dropout(x.float(), p=self.dr, training=True)
            x1 = F.relu(self.gcn[subtype](x1, edge_index_list[subtype]))
            output += x1
        output = F.normalize(output, dim=1, p=2)  # p=2 means l2-normalization
        del x
        del x1
        # torch.cuda.empty_cache()
        # print('edge type done:', edge_type)
        return output

class ConvLayerLocal_v2(torch.nn.Module):
    def __init__(self,conv_type,  in_channel, out_channel, bias, dr, edge_type, n_edge_subtype, n_drugs, n_genes):
        '''
        edge_subtype(int) = number of subtypes in an edge type e.g. number of cell lines for drug_drug edges.
        the difference btwn ConvLayerLocal and ConvLayerLocal_v2 is that it
        got rid of relu on each message passed from each neighbour. rather implement relu over neighbour sum.
        '''
        super(ConvLayerLocal_v2, self).__init__()
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
                self.gcn.append(conv_dict[conv_type](self.in_channel, self.out_channel, bias = bias))
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

        edge_index_list = train_pos_edges_dict[self.edge_type]
        for subtype in range(len(edge_index_list)):
            x1 = F.dropout(x.float(), p=self.dr, training=True)
            x1 = self.gcn[subtype](x1, edge_index_list[subtype])
            output += x1

        del x
        del x1
        return output

class ConvLayerLocal_v3(torch.nn.Module):
    def __init__(self,conv_type,  in_channel, out_channel, bias, dr, edge_type, n_edge_subtype, n_drugs, n_genes):
        '''
        edge_subtype(int) = number of subtypes in an edge type e.g. number of cell lines for drug_drug edges.
        the difference btwn ConvLayerLocal and ConvLayerLocal_v3 is that the later one ensures that while aggregating neighbours
        information of node u,  we consider node u's features only once, i.e. not mutiple times for mutilple cell lines or so.'''


        super(ConvLayerLocal_v3, self).__init__()
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
                self.gcn.append(conv_dict[conv_type](self.in_channel, self.out_channel, bias = bias, add_self_loops = False ))
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

        edge_index_list = train_pos_edges_dict[self.edge_type]
        for subtype in range(len(edge_index_list)):
            x1 = F.dropout(x.float(), p=self.dr, training=True)
            x1 = self.gcn[subtype](x1, edge_index_list[subtype])
            output += x1

        del x
        del x1
        return output

class ConvLayerGlobal(torch.nn.Module):  # in this function I have used one weight matrix for all drug_drug cell_lines
    def __init__(self, conv_type, in_channel, out_channel, bias, dr,edge_type, n_drugs, n_genes):
        super(ConvLayerGlobal, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.n_drugs = n_drugs
        self.n_genes = n_genes
        self.dr = dr
        self.edge_type = edge_type

        if edge_type in ['gene_gene', 'drug_drug']: #might change here to incorporate cellline spec weights in encoder
            self.gcn = conv_dict[conv_type](self.in_channel, self.out_channel, bias = bias, add_self_loops=True, cached=False)
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

        del x
        del x1
        # print('edge type done:', edge_type)
        return output


class BipartiteGCN(MessagePassing):
    r"""The graph convolutional operator from the `"Semi-supervised
    Classification with Graph Convolutional Networks"
    <https://arxiv.org/abs/1609.02907>`_ paper

    .. math::
        \mathbf{X}^{\prime} = \mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
        \mathbf{\hat{D}}^{-1/2} \mathbf{X} \mathbf{\Theta},

    where :math:`\mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}` denotes the
    adjacency matrix with inserted self-loops and
    :math:`\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij}` its diagonal degree matrix.
    The adjacency matrix can include other values than :obj:`1` representing
    edge weights via the optional :obj:`edge_weight` tensor.

    Its node-wise formulation is given by:

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{\Theta} \sum_{j}
        \frac{1}{\sqrt{\hat{d}_j \hat{d}_i}} \mathbf{x}_j

    with :math:`\hat{d}_i = 1 + \sum_{j \in \mathcal{N}(i)} e_{j,i}`, where
    :math:`e_{j,i}` denotes the edge weight from source node :obj:`i` to target
    node :obj:`j` (default: :obj:`1`)

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        improved (bool, optional): If set to :obj:`True`, the layer computes
            :math:`\mathbf{\hat{A}}` as :math:`\mathbf{A} + 2\mathbf{I}`.
            (default: :obj:`False`)
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the computation of :math:`\mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
            \mathbf{\hat{D}}^{-1/2}` on first execution, and will use the
            cached version for further executions.
            This parameter should only be set to :obj:`True` in transductive
            learning scenarios. (default: :obj:`False`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        normalize (bool, optional): Whether to add self-loops and apply
            symmetric normalization. (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    _cached_edge_index: Optional[Tuple[Tensor, Tensor]]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(self, in_channels: int, out_channels: int, adj_shape: tuple,
                 bias: bool = True,**kwargs):

        kwargs.setdefault('aggr', 'add')
        super(BipartiteGCN, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        # self.improved = improved
        # self.cached = cached
        # self.add_self_loops = add_self_loops
        # self.normalize = normalize

        # self._cached_edge_index = None
        # self._cached_adj_t = None

        self.adj_shape = adj_shape

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))
        self.adj_norm = None
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        utils.glorot(self.weight)
        utils.zeros(self.bias)
        # self._cached_edge_index = None
        # self._cached_adj_t = None

    def norm(self, edge_index):

        s = edge_index.size()[1]
        data = np.ones(s)
        edge_index_cpu = edge_index.cpu()
        adj = sp.csr_matrix((data, (edge_index_cpu[0], edge_index_cpu[1])), shape = self.adj_shape)
        adj = sp.coo_matrix(adj)
        rowsum = np.array(adj.sum(1))
        colsum = np.array(adj.sum(0))
        rowdegree_mat_inv = sp.diags(np.nan_to_num(np.power(rowsum, -0.5)).flatten())
        coldegree_mat_inv = sp.diags(np.nan_to_num(np.power(colsum, -0.5)).flatten())
        adj_normalized = rowdegree_mat_inv.dot(adj).dot(coldegree_mat_inv)
        adj_normalized = np.transpose(adj_normalized)
        adj_normalized = utils.np_sparse_to_sparse_tensor(adj_normalized).to(edge_index.device)
        return adj_normalized



    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        """"""
        x = torch.matmul(x, self.weight)

        if self.adj_norm==None:
            self.adj_norm = self.norm(edge_index)
        assert self.adj_norm.size()[1] == x.size()[0], 'problem in size: ' + \
                                    str(self.adj_norm.size()[1]) +'  '+ str(x.size()[0])
        out = torch.matmul(self.adj_norm, x)

        #write code for bias
        if self.bias is not None:
            out += self.bias

        return out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)
