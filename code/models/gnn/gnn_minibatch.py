from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import math
import copy
import scipy.sparse as sp
import random
from submodules.decagon.decagon.utility import preprocessing

np.random.seed(123)

class MinibatchHandler(object):

    def __init__(self, train_pos_edges_dict, batch_size, total_cell_lines):
        self.iter = 0
        # self.neg_fact = neg_fact
        self.total_cell_lines = total_cell_lines
        edge_types = train_pos_edges_dict.keys()
        self.edge_type_wise_n_batches = {edge_type: [] for edge_type in edge_types}
        self.edge_type_wise_next_batch = {edge_type: [] for edge_type in edge_types}
        for edge_type in edge_types:
            for i in range(len(train_pos_edges_dict[edge_type])):
                self.edge_type_wise_n_batches[edge_type].\
                    append(math.ceil(train_pos_edges_dict[edge_type][i].size()[1]/batch_size))
                self.edge_type_wise_next_batch[edge_type].append(0)


    def shuffle_train_edges(self, train_edges_dict):
        #train_edges_dict contains list of tensors for each edge type. e.g. a key:value pair => 'drug_drug':[edges_tensor_for_cell_line_1,\
        # edges_tensor_for_cell_line_2,...]; another key:value pair => 'gene_gene':[edges_tensor_for_gene_gene]
        for edge_type in train_edges_dict:
            for i in range(len(train_edges_dict[edge_type])):
                size_of_tensor = list(train_edges_dict[edge_type][i].size())
                r = [0,1]  #all the tensor has two rows, (row0= soruce, row1=target)
                c = torch.randperm(size_of_tensor[1])
                # print(r, c)
                # print()
                train_edges_dict[edge_type][i] = train_edges_dict[edge_type][i][r][:, c]
                # z = z[r][:, c]
        return train_edges_dict

    def next_minibatch(self):
        """Select a random edge type and a batch of edges of the same type"""
        while True:
            if self.iter % 4 == 0:
                # gene-gene relation
                self.current_edge_type = 'gene_gene'
                self.edge_sub_type = 0
            elif self.iter % 4 == 1:
                # gene-drug relation
                self.current_edge_type = 'target_drug'
                self.edge_sub_type = 0
            elif self.iter % 4 == 2:
                # drug-gene relation
                self.current_edge_type = 'drug_target'
                # self.current_edge_type = 'target_drug'
                self.edge_sub_type = 0
            else:
                # random cell line specific drug-drug relation
                self.current_edge_type = 'drug_drug'
                self.edge_sub_type =  np.random.choice(self.total_cell_lines)

            self.iter += 1
            #check if all the batches of selected edgetype are finished.
            if self.edge_type_wise_next_batch[self.current_edge_type][self.edge_sub_type]< \
                    self.edge_type_wise_n_batches[self.current_edge_type][self.edge_sub_type]:

                batch_num = self.edge_type_wise_next_batch[self.current_edge_type][self.edge_sub_type]
                self.edge_type_wise_next_batch[self.current_edge_type][self.edge_sub_type] += 1

                return self.current_edge_type, self.edge_sub_type, batch_num

    def next_minibatch_multippi(self):
        """Select a random edge type and a batch of edges of the same type"""
        while True:
            if self.iter % 4 == 0:
                # gene-gene relation
                self.current_edge_type = 'gene_gene'
                self.edge_sub_type = np.random.choice(self.total_cell_lines)
            elif self.iter % 4 == 1:
                # gene-drug relation
                self.current_edge_type = 'target_drug'
                self.edge_sub_type = 0
            elif self.iter % 4 == 2:
                # drug-gene relation
                self.current_edge_type = 'drug_target'
                # self.current_edge_type = 'target_drug'
                self.edge_sub_type = 0
            else:
                # random cell line specific drug-drug relation
                self.current_edge_type = 'drug_drug'
                self.edge_sub_type =  np.random.choice(self.total_cell_lines)

            self.iter += 1
            #check if all the batches of selected edgetype are finished.
            if self.edge_type_wise_next_batch[self.current_edge_type][self.edge_sub_type]< \
                    self.edge_type_wise_n_batches[self.current_edge_type][self.edge_sub_type]:

                batch_num = self.edge_type_wise_next_batch[self.current_edge_type][self.edge_sub_type]
                self.edge_type_wise_next_batch[self.current_edge_type][self.edge_sub_type] += 1

                return self.current_edge_type, self.edge_sub_type, batch_num

    # def next_minibatch_new(self):
    #     """Select a random edge type and a batch of edges of the same type"""
    #     while True:
    #         if self.iter % 4 == 0:
    #             # gene-gene relation
    #             self.current_edge_type = 'gene_gene'
    #             self.edge_sub_type = 0
    #         elif self.iter % 4 == 1:
    #             # gene-drug relation
    #             self.current_edge_type = 'target_drug'
    #             self.edge_sub_type = 0
    #         elif self.iter % 4 == 2:
    #             # drug-gene relation
    #             self.current_edge_type = 'drug_target'
    #             # self.current_edge_type = 'target_drug'
    #             self.edge_sub_type = 0
    #         else:
    #             # random cell line specific drug-drug relation
    #             self.current_edge_type = 'drug_drug'
    #             self.edge_sub_type = np.random.choice(self.total_cell_lines)
    #
    #         self.iter += 1
    #         #check if all the batches of selected edgetype are finished.
    #         if self.edge_type_wise_next_batch[self.current_edge_type][self.edge_sub_type]<\
    #             self.edge_type_wise_n_batches[self.current_edge_type][self.edge_sub_type]:
    #
    #             batch_num = self.edge_type_wise_next_batch[self.current_edge_type][self.edge_sub_type]
    #             self.edge_type_wise_next_batch[self.current_edge_type][self.edge_sub_type] += 1
    #
    #             return self.current_edge_type, self.edge_sub_type, batch_num


    def is_batch_finished(self):
        #returns true if every batch from every edge type is finished
        for edge_type in self.edge_type_wise_n_batches:
            for i in range(len(self.edge_type_wise_n_batches[edge_type])):
                total_batch = self.edge_type_wise_n_batches[edge_type][i]
                next_batch_no = self.edge_type_wise_next_batch[edge_type][i]
                if(total_batch>next_batch_no):
                    return False
        #if every edge type is finished with its batches then reset edge_type_wise_next_batch before\
        # returning batch_finished=true
        for edge_type in self.edge_type_wise_next_batch:
            for i in range(len(self.edge_type_wise_next_batch[edge_type])):
                self.edge_type_wise_next_batch[edge_type][i]=0
        return True

    def is_batch_finished_new(self):
        #returns true if all the 'drug_drug' batches are finished
        edge_type = 'drug_drug'

        for i in range(len(self.edge_type_wise_n_batches[edge_type])):
            total_batch = self.edge_type_wise_n_batches[edge_type][i]
            next_batch_no = self.edge_type_wise_next_batch[edge_type][i]
            if(total_batch>next_batch_no):
                return False
        #if every edge type is finished with its batches then reset edge_type_wise_next_batch before\
        # returning batch_finished=true
        for edge_type in self.edge_type_wise_next_batch:
            for i in range(len(self.edge_type_wise_next_batch[edge_type])):
                self.edge_type_wise_next_batch[edge_type][i]=0
        return True