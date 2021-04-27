import torch
from torch.nn import Parameter
import math
import numpy as np
import networkx as nx
import scipy.sparse as sp
import os
import pandas as pd

def np_sparse_to_sparse_tensor(adj_normalized):
    if not sp.isspmatrix_coo(adj_normalized):
        sparse_mx = adj_normalized.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col))
    values = sparse_mx.data
    shape = sparse_mx.shape
    adj_normalized = torch.sparse.FloatTensor(torch.LongTensor(coords), \
                                              torch.FloatTensor(values), torch.Size(shape))
    return adj_normalized

def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

def weight_matrix_glorot(in_channels, out_channels):
    """Create a weight variable with Glorot & Bengio (AISTATS 2010)
    initialization.
    """
    w = Parameter(torch.Tensor(in_channels, out_channels))
    stdv = math.sqrt(6.0 / (in_channels + in_channels))
    w.data.uniform_(-stdv, stdv)
    return w




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

def precision_at_k(y_true, y_score, k):
    df = pd.DataFrame({'true': y_true.tolist(), 'score': y_score.tolist()}).sort_values('score', ascending=False)
    df.reset_index(inplace=True)
    threshold = df.loc[k - 1]['score']
    df = df[df.score >= threshold]
    return df.true.sum() / df.shape[0]
