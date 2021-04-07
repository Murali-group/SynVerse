from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import copy
import scipy.sparse as sp
import random
from submodules.decagon.decagon.utility import preprocessing

np.random.seed(123)

class MinibatchHandler(object):

    def __init__(self, train_pos_edges_dict, train_neg_edges_dict, batch_size, neg_fact):
        self.iter = 0
        self.neg_fact = neg_fact

        self.edge_type_wise_n_batches = {}
        for edge_type in train_pos_edges_dict:
            self.edge_type_wise_n_batches[edge_type] = []
            for i in train_pos_edges_dict[edge_type]:
                self.edge_type_wise_n_batches[edge_type].\
                    append(int(train_pos_edges_dict[edge_type][i].size()[1]/batch_size))
        self.currently_remaining_n_batches = copy.deepcopy(self.edge_type_wise_n_batches)



    def shuffle_train_edges(train_edges_dict):
        #train_edges_dict contains list of tensors for each edge type. e.g. a key:value pair => 'drug_drug':[edges_tensor_for_cell_line_1,\
        # edges_tensor_for_cell_line_2,...]; another key:value pair => 'gene_gene':[edges_tensor_for_gene_gene]
        for edge_type in train_edges_dict:
            for i in range(len(train_edges_dict[edge_type])):
                size_of_tensor = train_edges_dict[edge_type][i].size()
                r = [0,1]  #all the tensor has two rows, (row0= soruce, row1=target)
                c = torch.randperm(size_of_tensor[1])
                train_edges_dict[edge_type][i] = train_edges_dict[edge_type][i][r][:c]
                # z = z[r][:, c]
        return train_edges_dict

    def next_minibatch(self):
        """Select a random edge type and a batch of edges of the same type"""
        while True:
            if self.iter % 4 == 0:
                # gene-gene relation
                self.current_edge_type = 'gene_gene'
            elif self.iter % 4 == 1:
                # gene-drug relation
                self.current_edge_type_idx = 'target_drug'
            elif self.iter % 4 == 2:
                # drug-gene relation
                self.current_edge_type_idx = 'drug_target'
            else:
                # random cell line specific drug-drug relation
                if len(self.freebatch_edge_types) > 0:
                    self.current_edge_type_idx = np.random.choice(self.freebatch_edge_types)
                else:
                    self.current_edge_type_idx = self.edge_type2idx[0, 0, 0]
                    self.iter = 0

            i, j, k = self.idx2edge_type[self.current_edge_type_idx]
            if self.batch_num[self.current_edge_type_idx] * self.batch_size \
                   <= len(self.train_edges[i,j][k]) - self.batch_size + 1:
                break
            else:
                if self.iter % 4 in [0, 1, 2]:
                    self.batch_num[self.current_edge_type_idx] = 0
                else:
                    self.freebatch_edge_types.remove(self.current_edge_type_idx)

        self.iter += 1
        start = self.batch_num[self.current_edge_type_idx] * self.batch_size
        self.batch_num[self.current_edge_type_idx] += 1
        batch_edges = self.train_edges[i,j][k][start: start + self.batch_size]
        neg_batch_edges = self.train_edges_false[i, j][k][start: start + self.batch_size]
        return self.batch_feed_dict(batch_edges, neg_batch_edges, self.current_edge_type_idx, placeholders)



































# class EdgeMinibatchIterator(object):
#     """ This minibatch iterator iterates over batches of sampled edges or
#     random pairs of co-occuring edges.
#     assoc -- numpy array with target edges
#     placeholders -- tensorflow placeholders object
#     batch_size -- size of the minibatches
#     """
#     def __init__(self, adj_mats, feat, edge_types, pos_drug_drug_validation_all_folds, neg_drug_drug_validation_all_folds,\
#                  non_drug_drug_validation_all_folds,neg_non_drug_drug_validation_all_folds, current_val_fold_no, batch_size=100):
#         self.adj_mats = adj_mats
#         self.feat = feat
#         self.edge_types = edge_types
#         self.batch_size = batch_size
#         # self.val_test_size = val_test_size
#         self.num_edge_types = sum(self.edge_types.values())
#
#         self.iter = 0
#         self.freebatch_edge_types = list(range(self.num_edge_types))
#         self.batch_num = [0]*self.num_edge_types
#         self.current_edge_type_idx = 0
#         self.edge_type2idx = {}
#         self.idx2edge_type = {}
#         r = 0
#         for i, j in self.edge_types:
#             for k in range(self.edge_types[i,j]):
#                 self.edge_type2idx[i, j, k] = r
#                 self.idx2edge_type[r] = i, j, k
#                 r += 1
#
#         self.train_edges = {edge_type: [None]*n for edge_type, n in self.edge_types.items()}
#         self.train_edges_false = {edge_type: [None] * n for edge_type, n in self.edge_types.items()}
#
#         self.val_edges = {edge_type: [None]*n for edge_type, n in self.edge_types.items()}
#         self.val_edges_false = {edge_type: [None] * n for edge_type, n in self.edge_types.items()}
#         # self.test_edges = {edge_type: [None]*n for edge_type, n in self.edge_types.items()}
#         # self.test_edges_false = {edge_type: [None]*n for edge_type, n in self.edge_types.items()}
#
#
#
#         self.adj_train = {edge_type: [None]*n for edge_type, n in self.edge_types.items()}
#         for i, j in self.edge_types:
#             for k in range(self.edge_types[i,j]):
#                 # print("Minibatch edge type:", "(%d, %d, %d)" % (i, j, k))
#                 self.new_mask_test_edges((i, j), k, pos_drug_drug_validation_all_folds,neg_drug_drug_validation_all_folds,\
#                                          non_drug_drug_validation_all_folds,neg_non_drug_drug_validation_all_folds, current_val_fold_no )
#                 # self.mask_test_edges((i, j), k, validation_fold)
#                 # print("Train edges=", "%04d" % len(self.train_edges[i,j][k]))
#                 # print("Val edges=", "%04d" % len(self.val_edges[i,j][k]))
#
#                 # print("Test edges=", "%04d" % len(self.test_edges[i,j][k]))
#
#         self.assert_train_val_non_overlapping()
#
#     def preprocess_graph(self, adj):
#         adj = sp.coo_matrix(adj)
#         if adj.shape[0] == adj.shape[1]:
#             adj_ = adj + sp.eye(adj.shape[0])
#             rowsum = np.array(adj_.sum(1))
#             degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
#             adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
#         else:
#             rowsum = np.array(adj.sum(1))
#             colsum = np.array(adj.sum(0))
#             rowdegree_mat_inv = sp.diags(np.nan_to_num(np.power(rowsum, -0.5)).flatten())
#             coldegree_mat_inv = sp.diags(np.nan_to_num(np.power(colsum, -0.5)).flatten())
#             adj_normalized = rowdegree_mat_inv.dot(adj).dot(coldegree_mat_inv).tocoo()
#         return preprocessing.sparse_to_tuple(adj_normalized)
#
#     def _ismember(self, a, b):
#         a = np.array(a)
#         b = np.array(b)
#         rows_close = np.all(a - b == 0, axis=1)
#         return np.any(rows_close)
#
#     def is_overlapping(self, data_1, data_2):
#         #data_1 and data_2 are 2D numpy array i.e. [[1,2],[2,4]]
#         data_1_set = set([(max(idx_1, idx_2), min(idx_1, idx_2)) for idx_1, idx_2 in data_1])
#         data_2_set = set([(max(idx_1, idx_2), min(idx_1, idx_2)) for idx_1, idx_2 in data_2])
#
#
#         # print('data_1, data_2', list(data_1_set)[0:5], list(data_2_set)[0:5])
#         common = data_1_set.intersection(data_2_set)
#         if len(common) == 0:
#             return False
#         else:
#             print(len(data_1_set), len(data_2_set), len(common))
#             return True
#
#     def assert_train_val_non_overlapping(self):
#         for i, j in self.edge_types:
#             for k in range(self.edge_types[i,j]):
#                 print(i,j,k)
#                 print(self.train_edges[(i,j)][k].shape,self.val_edges[(i,j)][k].shape, \
#                     self.train_edges[(i, j)][k].shape, self.val_edges_false[(i, j)][k].shape)
#
#                 assert self.is_overlapping (self.train_edges[(i, j)][k], self.val_edges[(i, j)][k]) == False,\
#                     'problem: pos train-vel overlap'
#                 assert self.is_overlapping(self.train_edges_false[(i, j)][k], self.val_edges_false[(i, j)][k]) == False,\
#                     'problem: neg train val overlap'
#
#
#     def new_mask_test_edges(self, edge_type, type_idx, pos_drug_drug_validation_all_folds, neg_drug_drug_validation_all_folds, \
#                             non_drug_drug_validation_all_folds, neg_non_drug_drug_validation_all_folds, current_val_fold):
#         pos_drug_drug_validation_fold = pos_drug_drug_validation_all_folds[current_val_fold]
#         neg_drug_drug_validation_fold  = neg_drug_drug_validation_all_folds[current_val_fold]
#         non_drug_drug_validation_fold = non_drug_drug_validation_all_folds[current_val_fold]
#         neg_non_drug_drug_validation_fold = neg_non_drug_drug_validation_all_folds[current_val_fold]
#         #create train, val_edges
#         edges_all, _, _ = preprocessing.sparse_to_tuple(self.adj_mats[edge_type][type_idx])
#         # print('investigate into edges all: ', edge_type, len(edges_all))
#         edges_set = set(map(tuple, edges_all))
#
#         #for drug-drug edges
#         if edge_type == (1,1):
#             #setup postive drug-drug edges
#             # for drug-drug edges only (x, y) is present ((y,x) is not )in passed cross-validation folds
#             cell_line_specific_pos_val_fold = \
#                 [[idx_1, idx_2] for idx_1, idx_2, idx_3 in pos_drug_drug_validation_fold if idx_3 == type_idx]
#             val_edges = [[idx_1, idx_2] for idx_1, idx_2 in cell_line_specific_pos_val_fold]
#             val_edges_set = set(map(tuple, val_edges))
#             val_edges = np.array(val_edges)
#             num_val = val_edges.shape[0]
#
#
#             #setup negative drug-drug edges
#             cell_line_specific_neg_val_fold = \
#                 [[idx_1, idx_2] for idx_1, idx_2, idx_3 in neg_drug_drug_validation_fold if idx_3 == type_idx]
#             val_edges_false = [[idx_1, idx_2] for idx_1, idx_2 in cell_line_specific_neg_val_fold]
#
#             val_edges_false = np.array(val_edges_false)
#             # self.val_edges_false[edge_type][type_idx] = val_edges_false
#
#         #for other type of edges e.g. drug-target, target-target
#         else:
#             #for gene-gene edges both (x,y) and (y,x) pairs are present in passed cross-validation folds
#             val_edges = np.array(non_drug_drug_validation_fold[edge_type]) #list of tuples
#             val_edges_set = set(map(tuple,val_edges))
#             num_val = val_edges.shape[0]
#
#             # setup negative samples as well
#             val_edges_false = np.array(neg_non_drug_drug_validation_fold[edge_type])
#             # self.val_edges_false[edge_type][type_idx] = val_edges_false
#
#         #positive train edges set
#         train_edges_set = edges_set.difference(val_edges_set)
#         train_edges = [[idx_1,idx_2] for idx_1,idx_2 in train_edges_set]
#         train_edges = np.array(train_edges)
#
#         #negative train edges set
#         train_edges_false_set = set()
#         # train_folds = list(range(len(neg_drug_drug_validation_all_folds))) - [current_val_fold]
#         if edge_type==(1,1):
#             for i in range(len(neg_drug_drug_validation_all_folds)):
#                 if i != current_val_fold:
#                     # neg_drug_drug_validation_all_folds[i] is a list of tuples (drug1_idx, drug2_idx, cell_line_idx)
#                     train_edges_false_set = train_edges_false_set.union(set(neg_drug_drug_validation_all_folds[i]))
#             # keep only the tuples where cell_line_idx == current cell_line index under consideration i.e. type_idx
#             train_edges_false = [[idx_1, idx_2] for idx_1, idx_2, idx_3 in train_edges_false_set if idx_3 == type_idx]
#             train_edges_false = np.array(train_edges_false)
#             # self.train_edges_false[edge_type][type_idx] = train_edges_false
#         else:
#             for i in range(len(neg_non_drug_drug_validation_all_folds)):
#                 if i != current_val_fold:
#                     # neg_drug_drug_validation_all_folds[i] is a list of tuples (drug1_idx, drug2_idx, cell_line_idx)
#                     train_edges_false_set = train_edges_false_set.union\
#                         (set([(idx_1, idx_2) for idx_1, idx_2 in neg_non_drug_drug_validation_all_folds[i][edge_type]]))
#             train_edges_false = np.array(list(train_edges_false_set))
#
#
#         #rebuild adjacency matrices
#         data = np.ones(train_edges.shape[0])
#         adj_train = sp.csr_matrix(
#             (data, (train_edges[:, 0], train_edges[:, 1])),
#             shape=self.adj_mats[edge_type][type_idx].shape)
#         self.adj_train[edge_type][type_idx] = self.preprocess_graph(adj_train)
#
#         #set values
#         self.train_edges[edge_type][type_idx] = train_edges
#         self.val_edges[edge_type][type_idx] = val_edges
#         self.val_edges_false[edge_type][type_idx] = val_edges_false
#         self.train_edges_false[edge_type][type_idx] = train_edges_false
#
#         # print('decagon minibatch: train edges, train edges false')
#         # print(edge_type)
#         # print(type(train_edges), train_edges.shape)
#         # print(type(train_edges_false), train_edges_false.shape)
#         # print(type(val_edges), val_edges.shape)
#         # print(type(val_edges_false), val_edges_false.shape)
#
#
#
#     def end(self):
#         finished = len(self.freebatch_edge_types) == 0
#         return finished
#
#     def update_feed_dict(self, feed_dict, dropout, placeholders):
#         # construct feed dictionary
#         feed_dict.update({
#             placeholders['adj_mats_%d,%d,%d' % (i,j,k)]: self.adj_train[i,j][k]
#             for i, j in self.edge_types for k in range(self.edge_types[i,j])})
#         feed_dict.update({placeholders['feat_%d' % i]: self.feat[i] for i, _ in self.edge_types})
#         feed_dict.update({placeholders['dropout']: dropout})
#
#         return feed_dict
#
#     def batch_feed_dict(self, batch_edges, neg_batch_edges, batch_edge_type, placeholders):
#         feed_dict = dict()
#         feed_dict.update({placeholders['batch']: batch_edges})
#         feed_dict.update({placeholders['neg_batch']: neg_batch_edges})
#         feed_dict.update({placeholders['batch_edge_type_idx']: batch_edge_type})
#         feed_dict.update({placeholders['batch_row_edge_type']: self.idx2edge_type[batch_edge_type][0]})
#         feed_dict.update({placeholders['batch_col_edge_type']: self.idx2edge_type[batch_edge_type][1]})
#
#         return feed_dict
#
#     def next_minibatch_feed_dict(self, placeholders):
#         """Select a random edge type and a batch of edges of the same type"""
#         while True:
#             if self.iter % 4 == 0:
#                 # gene-gene relation
#                 self.current_edge_type_idx = self.edge_type2idx[0, 0, 0]
#             elif self.iter % 4 == 1:
#                 # gene-drug relation
#                 self.current_edge_type_idx = self.edge_type2idx[0, 1, 0]
#             elif self.iter % 4 == 2:
#                 # drug-gene relation
#                 self.current_edge_type_idx = self.edge_type2idx[1, 0, 0]
#             else:
#                 # random cell line specific drug-drug relation
#                 if len(self.freebatch_edge_types) > 0:
#                     self.current_edge_type_idx = np.random.choice(self.freebatch_edge_types)
#                 else:
#                     self.current_edge_type_idx = self.edge_type2idx[0, 0, 0]
#                     self.iter = 0
#
#             i, j, k = self.idx2edge_type[self.current_edge_type_idx]
#             if self.batch_num[self.current_edge_type_idx] * self.batch_size \
#                    <= len(self.train_edges[i,j][k]) - self.batch_size + 1:
#                 break
#             else:
#                 if self.iter % 4 in [0, 1, 2]:
#                     self.batch_num[self.current_edge_type_idx] = 0
#                 else:
#                     self.freebatch_edge_types.remove(self.current_edge_type_idx)
#
#         self.iter += 1
#         start = self.batch_num[self.current_edge_type_idx] * self.batch_size
#         self.batch_num[self.current_edge_type_idx] += 1
#         batch_edges = self.train_edges[i,j][k][start: start + self.batch_size]
#         neg_batch_edges = self.train_edges_false[i, j][k][start: start + self.batch_size]
#         return self.batch_feed_dict(batch_edges, neg_batch_edges, self.current_edge_type_idx, placeholders)
#
#     def num_training_batches(self, edge_type, type_idx):
#         return len(self.train_edges[edge_type][type_idx]) // self.batch_size + 1
#
#     def val_feed_dict(self, edge_type, type_idx, placeholders, size=None): # Nure: no usage found for this function
#         edge_list = self.val_edges[edge_type][type_idx]
#         if size is None:
#             return self.batch_feed_dict(edge_list, edge_type, placeholders)
#         else:
#             ind = np.random.permutation(len(edge_list))
#             val_edges = [edge_list[i] for i in ind[:min(size, len(ind))]]
#             return self.batch_feed_dict(val_edges, edge_type, placeholders)
#
#     def shuffle(self):
#         """ Re-shuffle the training set.
#             Also reset the batch number.
#         """
#         for edge_type in self.edge_types:
#             for k in range(self.edge_types[edge_type]):
#                 self.train_edges[edge_type][k] = np.random.permutation(self.train_edges[edge_type][k])
#                 self.train_edges_false[edge_type][k] = np.random.permutation(self.train_edges_false[edge_type][k])
#                 self.batch_num[self.edge_type2idx[edge_type[0], edge_type[1], k]] = 0
#
#         self.current_edge_type_idx = 0
#         self.freebatch_edge_types = list(range(self.num_edge_types))
#         self.freebatch_edge_types.remove(self.edge_type2idx[0, 0, 0])
#         self.freebatch_edge_types.remove(self.edge_type2idx[0, 1, 0])
#         self.freebatch_edge_types.remove(self.edge_type2idx[1, 0, 0])
#         self.iter = 0
#
