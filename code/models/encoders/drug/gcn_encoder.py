import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from torch_geometric.nn import global_max_pool as gmp

import os
from torch_geometric.data import InMemoryDataset
import torch
torch.set_default_dtype(torch.float32)


class GNN_data(InMemoryDataset):
    def __init__(self, root='/tmp', dataset='_drug',
                 data_list=None, transform=None,
                 pre_transform=None):

        #root is required for save preprocessed data, default is '/tmp'
        super(GNN_data, self).__init__(root, transform, pre_transform)
        self.dataset = dataset
        self.data_list = data_list
        self.data, self.slices, self.batch = self.process()


    @property
    def raw_file_names(self):
        pass
        #return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return [self.dataset + '.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    # Return: PyTorch-Geometric format processed data
    def process(self):
        if self.pre_filter is not None:
            self.data_list = [data for data in self.data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            self.data_list = [self.pre_transform(data) for data in self.data_list]
        data, slices = self.collate(self.data_list)

        batch = [] #create batch to keep track of which atom belongs to which drug
        slice_drug = list(slices['x'].numpy())
        for i in range(len(slice_drug) - 1):
            # Append the current value (i) for the range between two start indices
            batch.extend([i] * (slice_drug[i + 1] - slice_drug[i]))
        batch = torch.LongTensor(batch)
        return data, slices, batch



class GCN_Encoder(nn.Module):
    def __init__(self, input_size, config):
        '''
        :param input_size: atomic feature size
        :param config: gnn specific config
        '''
        super().__init__()
        # self.relu = nn.ReLU()
        # self.dropout = nn.Dropout(config['gnn_dropout'])
        self.batch_norm = config['batch_norm']
        #add convolutional layers
        gnn_n_layers = config['gnn_num_layers']
        gnn_layer_dims = []
        for i in range(gnn_n_layers):
            gnn_layer_dims.append(config[f'gnn_{i}'])

        self.gnn_layers = nn.ModuleList()
        for i in range(len(gnn_layer_dims)):
            self.gnn_layers.append(GCNConv(input_size if i==0 else gnn_layer_dims[i - 1], gnn_layer_dims[i], add_self_loops=True))

        #add feed forward layers to apply on graph embedding after max pooling
        n_ff_hid_layers = config['ff_num_layers']
        ff_layers  = []
        for i in range(n_ff_hid_layers):
            ff_layers.append(config[f'ff_{i}'])

        self.ff_layers = nn.ModuleList()
        for i in range(len(ff_layers)): #last_layer_of_GNN => input_first_feed_forward_layers.
            self.ff_layers.append(nn.Linear(gnn_layer_dims[gnn_n_layers-1]if i==0 else ff_layers[i-1], ff_layers[i]))
            self.ff_layers.append(nn.ReLU())
            if self.batch_norm:
                self.ff_layers.append(nn.BatchNorm1d(ff_layers[i]))
            self.ff_layers.append(nn.Dropout(config['gnn_dropout']))

        self.out_dim = ff_layers[n_ff_hid_layers-1]


    def forward(self, data_list, device):
        gnn_data = GNN_data(data_list=data_list)
        x, edge_index, batch = gnn_data.x, gnn_data.edge_index, gnn_data.batch

        x = x.to(device)
        edge_index = edge_index.to(device)
        batch = batch.to(device)

        for layer in self.gnn_layers:
            x = layer(x, edge_index)
            x = F.relu(x)

        x = gmp(x, batch)  # global max pooling

        for layer in self.ff_layers:
            x = layer(x)
            # x = self.relu(x)
            # x = self.dropout(x)

        return x