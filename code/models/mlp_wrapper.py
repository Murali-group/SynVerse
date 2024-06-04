'''
Define the mlp model here.
'''
import numpy as np
import torch
import torch.nn as nn
torch.set_default_dtype(torch.float64)
from models.decoders.mlp import MLP
class MLP_wrapper(nn.Module):
    def __init__(self, input_size, config):
        super().__init__()
        self.chosen_config = config
        self.mlp = MLP(input_size, config)

    def prepare_drug_feat(self, drug_feat):
        dfeats_array = [drug_feat[feature] for feature in drug_feat]
        drug_X = np.concatenate(dfeats_array, axis=1)
        return drug_X

    def prepare_cell_feat(self, cell_line_feat):
        cfeats_array = [cell_line_feat[feature] for feature in cell_line_feat]
        cell_line_X = np.concatenate(cfeats_array, axis=1)
        return cell_line_X

    def concat_feat(self, batch_triplets, drug_X, cell_X):
        '''
        :param x: tensor: triplets for a batch
        :param drug_feat: dict: key=feature name, value = drug feature as numpy array where
                          row i is the feature for drug idx with i.
        :param cell_line_feat: dict: key=feature name, value = cell line feature as numpy array where
                          row i is the feature for cell line idx with i.
        :return: concat features of drug_pair and cell line appearing in each triplet
            in the current batch and return it.
        '''

        drug1s = batch_triplets[:, 0].flatten()
        drug2s = batch_triplets[:, 1].flatten()
        cell_lines = batch_triplets[:, 2].flatten()
        # Use indexing to fetch all necessary features directly
        source_features = drug_X[drug1s]
        target_features = drug_X[drug2s]
        edge_features = cell_X[cell_lines]
        # Concatenate the features along the second axis (column-wise concatenation)
        # concatenation: source-target-edge_type
        # concatenated_features = np.concatenate([source_features, target_features, edge_features], axis=1)
        mlp_ready_feats = torch.tensor(np.concatenate([source_features, target_features, edge_features], axis=1))
        return mlp_ready_feats

    def forward(self, batch_triplets, drug_feat, cell_line_feat, device):
        drug_X= self.prepare_drug_feat(drug_feat)
        cell_X = self.prepare_cell_feat(cell_line_feat)
        x = self.concat_feat(batch_triplets, drug_X, cell_X)
        x = x.to(device)
        x = self.mlp(x)
        # TODO: remove after making sure nn.dataparalle is working.
        # print("Inside: input size", batch_triplets.size())
        return x

    def number_of_parameters(self):
        return (sum(p.numel() for p in self.parameters() if p.requires_grad))

