'''
Define the mlp model here.
'''
import torch
import torch.nn as nn

torch.set_default_dtype(torch.float64)
import time
from abc import ABC, abstractmethod
import numpy as np

class Base_Model(nn.Module):
    def __init__(self):
        super(Base_Model, self).__init__()

    @abstractmethod
    def forward(self, batch_triplets, drug_feat, cell_line_feat, device):
        pass

    def number_of_parameters(self):
        return (sum(p.numel() for p in self.parameters() if p.requires_grad))

