import copy

from models.mlp_wrapper import *
from models.model_utils import *
from models.runner import  Runner

import torch.optim as optim
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, TensorDataset

from models.mlp_worker import MLPWorker

class MLP_runner (Runner):
    def __init__(self, train_val_triplets_df, train_idx, val_idx, dfeat_dict, cfeat_dict,
                 dfeat_dim_dict, cfeat_dim_dict, out_file_prefix, params, model_info, device, **kwargs):
        super().__init__(train_val_triplets_df, train_idx, val_idx, dfeat_dict,
                         cfeat_dict, dfeat_dim_dict, cfeat_dim_dict, out_file_prefix, params, model_info, device, **kwargs)

        # save the name of the worker class applicable an attribute and later use in for finding best hyperparam.
        self.worker_cls = MLPWorker
        self.input_dim = self.get_input_dim()

    def get_input_dim(self):
        drug_dim=0
        cell_line_dim=0
        for feat in self.drug_feat:
            drug_dim += self.drug_feat[feat].shape[1]
        for feat in self.cell_line_feat:
            cell_line_dim += self.cell_line_feat[feat].shape[1]

        drug_pair_dim =drug_dim*2
        input_dim = drug_pair_dim + cell_line_dim
        return input_dim

    def init_model(self, config):
        model = MLP_wrapper(self.input_dim, config)
        ## Wrap the model for parallel processing if multiple gpus are available
        # if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        #     print(f"Using {torch.cuda.device_count()} GPUs!")
        #     model = nn.DataParallel(model)

        model.to(self.device)

        criterion = nn.MSELoss()
        if config['optimizer'] == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=config['lr'])
        else:
            optimizer = optim.SGD(model.parameters(), lr=config['lr'], momentum=config['sgd_momentum'])
        print('Model initialization done')
        return model, optimizer, criterion





