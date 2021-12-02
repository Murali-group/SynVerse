
import numpy as np

import torch
import torch.nn as nn
from torch.nn import Parameter


import models.gnn.gnn_utils as utils

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FeedforwardNeuralNetModel(nn.Module):
    def __init__(self, input_dim, hidden_layer_dims, output_dim):
        super(FeedforwardNeuralNetModel, self).__init__()
        self.hidden_layer_dims = hidden_layer_dims
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_layer_dims[0]))
        for i in range(1, len(self.hidden_layer_dims)):
            self.layers.append(nn.Linear(hidden_layer_dims[i-1], hidden_layer_dims[i]))
        self.layers.append(nn.Linear(hidden_layer_dims[-1], output_dim))

        # Non-linearity
        # self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()


    def forward(self, x):
        out = self.layers[0](x)
        for i in range(1, len(self.layers)):
            # Non-linearity
            out = self.relu(out)

            #can use batch norm here.
            out = self.layers[i](out)
        return out


class NNDecoder(torch.nn.Module):
    #this decoder is applicable for only drug_drug edge prediction
    def __init__(self, edge_type, w_dim, hidden_layer_setup, gene_expression_df):
        super(NNDecoder, self).__init__()
        self.edge_type = edge_type
        self.gene_expression_df = gene_expression_df

        self.hidden_layer_setup = hidden_layer_setup
        self.out_layer = 1
        self.input_dim = w_dim
        self.feed_forward_model = FeedforwardNeuralNetModel \
            (w_dim, self.hidden_layer_setup, self.out_layer).to(dev)

    def forward(self, z, batch_edges, edge_sub_type_idx, is_sigmoid=True):

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

            if concatenated_drug_drug_genex_embedding.size()[1] != self.input_dim:
                print('problem with input dim')
            assert concatenated_drug_drug_genex_embedding.size()[1]==self.input_dim, print('problem with input dim')
            #now feed the input to nn model
            score = self.feed_forward_model(concatenated_drug_drug_genex_embedding)
            score = torch.flatten(score)
            return torch.sigmoid(score) if is_sigmoid else score



class NNDecoder_nogenex(torch.nn.Module):
    #this decoder is applicable for only drug_drug edge prediction
    def __init__(self, edge_type, w_dim, hidden_layer_setup):
        super(NNDecoder_nogenex, self).__init__()
        self.edge_type = edge_type
        self.hidden_layer_setup = hidden_layer_setup
        self.out_layer = 1
        self.feed_forward_model = FeedforwardNeuralNetModel \
            (w_dim, self.hidden_layer_setup, self.out_layer).to(dev)

    def forward(self, z, batch_edges,edge_sub_type_idx, is_sigmoid=True):

        if self.edge_type == 'drug_drug':
            z1 = z['drug']
            z2 = z['drug']
            drug1_embed = z1[batch_edges[0]].to(dev)
            drug2_embed = z2[batch_edges[1]].to(dev)
            concatenated_drug_drug_embedding =\
                torch.cat((drug1_embed, drug2_embed), axis=1).type(torch.FloatTensor).to(dev)

            # print('size: ', concatenated_drug_drug_genex_embedding.size())

            #now feed the input to nn model
            score = self.feed_forward_model(concatenated_drug_drug_embedding)
            score = torch.flatten(score)
            return torch.sigmoid(score) if is_sigmoid else score





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



    def forward(self, z, batch_edges, edge_sub_type_idx, is_sigmoid=True):
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

        del product_1
        del product_2
        return torch.sigmoid(score) if is_sigmoid else score

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


    def forward(self, z, batch_edges, edge_sub_type_idx, is_sigmoid=True):
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

        del product_1
        del product_2
        del product_3
        del product_4
        return torch.sigmoid(score) if is_sigmoid else score

    # def reset_parameters(self):
    #     self.weight.data.normal_(std=1 / np.sqrt(self.in_dim))
