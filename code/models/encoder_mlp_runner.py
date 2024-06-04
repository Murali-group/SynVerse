from models.encoder_mlp_wrapper import *
from models.model_utils import *
from models.runner import  Runner
import torch.optim as optim
from models.encoder_mlp_worker import Encode_MLPWorker

class Encode_MLP_runner (Runner):
    def __init__(self, train_val_triplets_df, train_idx, val_idx, dfeat_dict, cfeat_dict,
                 dfeat_dim_dict, cfeat_dim_dict,
                 out_file_prefix, params,model_info, device, **kwargs):

        super().__init__(train_val_triplets_df, train_idx, val_idx, dfeat_dict,
            cfeat_dict,dfeat_dim_dict, cfeat_dim_dict, out_file_prefix, params,model_info, device, **kwargs)

        self.worker_cls = Encode_MLPWorker
        self.drug_encoder_info = model_info['drug_encoder']
        self.cell_encoder_info = model_info['cell_encoder']

    def init_model(self, config):
        model = Encoder_MLP_wrapper(self.drug_encoder_info, self.cell_encoder_info, self.dfeat_dim_dict, self.cfeat_dim_dict, config).to(self.device)
        ## Wrap the model for parallel processing if multiple gpus are available
        # if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        #     print(f"Using {torch.cuda.device_count()} GPUs!")
        #     model = nn.DataParallel(model)
        #setup loss
        criterion = nn.MSELoss()
        #setup optimizer
        if config is not None:
            if config['optimizer'] == 'Adam':
                optimizer = optim.Adam(model.parameters(), lr=config['lr'])
            else:
                optimizer = optim.SGD(model.parameters(), lr=config['lr'], momentum=config['sgd_momentum'])
        else:
            optimizer = optim.Adam(model.parameters(), lr=0.00001)

        print('Model initialization done')
        return model, optimizer, criterion




