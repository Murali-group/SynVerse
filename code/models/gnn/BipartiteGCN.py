from typing import Optional, Tuple
from torch_geometric.typing import Adj, OptTensor, PairTensor

import torch
from torch import Tensor
from torch.nn import Parameter
from torch_scatter import scatter_add
from torch_sparse import SparseTensor, matmul, fill_diag, sum, mul
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes
import scipy.sparse as sp
import numpy as np
import math

import models.gnn.gnn_utils as utils

def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)


def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)

#
#
# def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
#              add_self_loops=True, multinode=False, dtype=None):
#
#     fill_value = 2. if improved else 1.
#     num_nodes = maybe_num_nodes(edge_index, num_nodes)
#
#     if edge_weight is None:
#         edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
#                                  device=edge_index.device)
#
#     if add_self_loops:
#         edge_index, tmp_edge_weight = add_remaining_self_loops(
#             edge_index, edge_weight, fill_value, num_nodes)
#         assert tmp_edge_weight is not None
#         edge_weight = tmp_edge_weight
#
#     row, col = edge_index[0], edge_index[1]
#
#     if not multinode:
#         deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
#         deg_inv_sqrt = deg.pow_(-0.5)
#         deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
#         return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
#     else:
#         #add different normalization code for graph where nodes connected by an edge are different.




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
        glorot(self.weight)
        zeros(self.bias)
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
