import os
from itertools import islice
import sys
import numpy as np
from math import sqrt
from scipy import stats
from torch_geometric.data import InMemoryDataset, DataLoader
from torch_geometric import data as DATA
import torch

class GNN_data(InMemoryDataset):
    def __init__(self, root='/tmp', dataset='_drug',
                 data_list=None, xt=None, y=None, xt_featrue=None, transform=None,
                 pre_transform=None, smile_graph=None):

        #root is required for save preprocessed data, default is '/tmp'
        super(GNN_data, self).__init__(root, transform, pre_transform)
        # benchmark dataset, default = 'davis'
        self.dataset = dataset
        # if os.path.isfile(self.processed_paths[0]):
        #     print('Pre-processed data found: {}, loading ...'.format(self.processed_paths[0]))
        #     self.data, self.slices = torch.load(self.processed_paths[0])
        # else:
        # print('Pre-processed data {} not found, doing pre-processing...'.format(self.processed_paths[0]))
        self.data, self.slices, self.batch = self.process(data_list)
        # self.data, self.slices = torch.load(self.processed_paths[0])


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

    def get_cell_feature(self, cellId, cell_features):
        for row in islice(cell_features, 0, None):
            if cellId in row[0]:
                return row[1:]
        return False

    # Customize the process method to fit the task of drug-target affinity prediction
    # Inputs:
    # XD - list of SMILES, XT: list of encoded target (categorical or one-hot),
    # Y: list of labels (i.e. affinity)
    # Return: PyTorch-Geometric format processed data
    def process(self, data_list):
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        data, slices = self.collate(data_list)
        # save preprocessed data:
        # torch.save((data, slices), self.processed_paths[0])

        batch = [] #create batch to keep track of which atom belongs to which drug
        slice_drug = list(slices['x'].numpy())
        for i in range(len(slice_drug) - 1):
            # Append the current value (i) for the range between two start indices
            batch.extend([i] * (slice_drug[i + 1] - slice_drug[i]))
        batch = torch.LongTensor(batch)
        return data, slices, batch