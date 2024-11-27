try:
    import torch
    import torch.utils.data
    import torch.nn as nn
    import torch.nn.functional as F
except:
    raise ImportError("For this example you need to install pytorch.")
from torch.utils.data import DataLoader, Subset

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from hpbandster.core.worker import Worker
import logging
logging.basicConfig(level=logging.DEBUG)
import time

import torch
torch.set_default_dtype(torch.float32)

class Encode_MLPWorker(Worker):
    # def __init__(self, train_val_dataset, train_idx, val_idx, input_dim=None, batch_size=4096,
    #             check_freq=2, tolerance=10,n_folds=5,is_wandb=False, device='cpu',sleep_interval=0, **kwargs):
    def __init__(self, runner_instance, sleep_interval=0, **kwargs):
        super().__init__(**kwargs)
        self.runner = runner_instance
        self.sleep_interval = sleep_interval

    def compute(self, config, budget, working_directory, *args, **kwargs):
        """
        Simple example for a compute function using a feed forward network.
        It is trained on the MNIST dataset.
        The input parameter "config" (dictionary) contains the sampled configurations passed by the bohb optimizer
        """

        val_losses = {}
        req_epochs = {}

        for fold in range(self.runner.n_folds):
            print('FOLD: ', fold)
            f = open(self.runner.log_file, 'a')
            f.write(f'Fold {fold}\n')
            f.close()

            fold_train_idx = self.runner.train_idx[fold]
            fold_val_idx = self.runner.val_idx[fold]
            train_subsampler = Subset(self.runner.triplets_scores_dataset, fold_train_idx)
            val_subsampler = Subset(self.runner.triplets_scores_dataset, fold_val_idx)

            train_loader = DataLoader(train_subsampler, batch_size=self.runner.batch_size, shuffle=True)
            val_loader = DataLoader(val_subsampler, batch_size=self.runner.batch_size, shuffle=False)

            model, optimizer, criterion = self.runner.init_model(config)

            best_model_state, val_losses[fold],_, req_epochs[fold] = self.runner.train_model(model, optimizer, criterion, train_loader,
                budget, self.runner.check_freq, self.runner.tolerance, self.runner.is_wandb, self.runner.device, early_stop=True, val_loader=val_loader)

            time.sleep(self.sleep_interval)

        avg_val_loss = sum(val_losses.values())/self.runner.n_folds
        max_epochs = max(req_epochs.values())

        return ({
                'loss': avg_val_loss, # remember: HpBandSter always minimizes!
                'n_epochs': max_epochs,
                'info': {
                        'validation loss': avg_val_loss,
                        }
        })

    @staticmethod
    def get_configspace(model_params):
        """
        It builds the configuration space with the needed hyperparameters.
        It is easily possible to implement different types of hyperparameters.
        Beside float-hyperparameters on a log scale, it is also able to handle categorical input parameter.
        :return: ConfigurationsSpace-Object
        """
        cs = CS.ConfigurationSpace()

        #**************************************************** DECODER ******************************************************
        if model_params['decoder']['name'] =='MLP':
            final_mlp_params = model_params['decoder']['hp_range']
            # ****************** GENERAL configurations ***********************************************
            lr = CSH.UniformFloatHyperparameter('lr', lower=final_mlp_params['lr'][0], upper=final_mlp_params['lr'][1], default_value=1e-4, log=True)
            optimizer = CSH.CategoricalHyperparameter('optimizer', final_mlp_params['optimizer'])
            sgd_momentum = CSH.UniformFloatHyperparameter('sgd_momentum', lower=final_mlp_params.get('sgd_momentum',[0,0])[0],
                            upper=final_mlp_params.get('sgd_momentum',[0.99,0.99])[1], default_value=0.9, log=False)
            cs.add_hyperparameters([lr, optimizer, sgd_momentum])
            # The hyperparameter sgd_momentum will be used,if the configuration
            # contains 'SGD' as optimizer.
            cond = CS.EqualsCondition(sgd_momentum, optimizer, 'SGD')
            cs.add_condition(cond)

            #********************* Configurations for synergy predicting MLP layer **********************

            num_hid_layers = CSH.UniformIntegerHyperparameter('num_hid_layers',
                                                              lower=final_mlp_params['num_hid_layers'][0],
                                                              upper=final_mlp_params['num_hid_layers'][1],
                                                              default_value=2)

            hid_0 = CSH.CategoricalHyperparameter('hid_0',final_mlp_params['hid_0'])
            hid_1 = CSH.CategoricalHyperparameter('hid_1', final_mlp_params['hid_1'])
            hid_2 = CSH.CategoricalHyperparameter('hid_2', final_mlp_params['hid_2'])
            cs.add_hyperparameters([num_hid_layers, hid_0, hid_1, hid_2])
            # add conditions so that hid_2 will be considered when num_hid_layers>1 and so on.
            cond = CS.GreaterThanCondition(hid_1, num_hid_layers, 1)
            cs.add_condition(cond)
            cond = CS.GreaterThanCondition(hid_2, num_hid_layers, 2)
            cs.add_condition(cond)
            in_dropout_rate = CSH.UniformFloatHyperparameter('in_dropout_rate', lower=final_mlp_params['in_dropout_rate'][0],
                                    upper=final_mlp_params['in_dropout_rate'][1], default_value=0.5,log=False)
            hid_dropout_rate = CSH.UniformFloatHyperparameter('hid_dropout_rate', lower=final_mlp_params['hid_dropout_rate'][0],
                                    upper=final_mlp_params['hid_dropout_rate'][1], default_value=0.5,log=False)

            cs.add_hyperparameters([in_dropout_rate, hid_dropout_rate])



        #************************ ENCODER specific configurations **************************************
        for drug_encoder in model_params['drug_encoder']:
            encoder_params = drug_encoder.get('hp_range')
            if (drug_encoder['name'] == 'GCN'):
                batch_norm = CSH.CategoricalHyperparameter('batch_norm', encoder_params['batch_norm'])
                cs.add_hyperparameters([batch_norm])

                gnn_num_layers = CSH.UniformIntegerHyperparameter('gnn_num_layers',
                                                                  lower=encoder_params['gnn_num_layers'][0],
                                                                  upper=encoder_params['gnn_num_layers'][1],
                                                                  default_value=2)
                gnn_0 = CSH.CategoricalHyperparameter('gnn_0', encoder_params['gnn_0'])
                gnn_1 = CSH.CategoricalHyperparameter('gnn_1', encoder_params['gnn_1'])
                gnn_2 = CSH.CategoricalHyperparameter('gnn_2', encoder_params['gnn_2'])
                cs.add_hyperparameters([gnn_num_layers, gnn_0, gnn_1, gnn_2])
                # add conditions so that hid_2 will be considered when num_hid_layers>1 and so on.
                cond = CS.GreaterThanCondition(gnn_1, gnn_num_layers, 1)
                cs.add_condition(cond)
                cond = CS.GreaterThanCondition(gnn_2, gnn_num_layers, 2)
                cs.add_condition(cond)
                #needs a/multiple feed forward layer as well
                ff_num_layers = CSH.UniformIntegerHyperparameter('ff_num_layers',
                                                                 lower=encoder_params['ff_num_layers'][0],
                                                                 upper=encoder_params['ff_num_layers'][1],
                                                                 default_value=2)
                ff_0 = CSH.CategoricalHyperparameter('ff_0', encoder_params['ff_0'])
                ff_1 = CSH.CategoricalHyperparameter('ff_1', encoder_params['ff_1'])
                ff_2 = CSH.CategoricalHyperparameter('ff_2', encoder_params['ff_2'])

                cs.add_hyperparameters([ff_num_layers, ff_0, ff_1, ff_2])
                # add conditions so that hid_2 will be considered when num_hid_layers>1 and so on.
                cond = CS.GreaterThanCondition(ff_1, ff_num_layers, 1)
                cs.add_condition(cond)
                cond = CS.GreaterThanCondition(ff_2, ff_num_layers, 2)
                cs.add_condition(cond)
                dropout = CSH.UniformFloatHyperparameter('gnn_dropout', lower=encoder_params['gnn_dropout'][0],
                                        upper=encoder_params['gnn_dropout'][1], default_value=0.5, log=False)
                cs.add_hyperparameters([dropout])

            if (drug_encoder['name']=='Transformer'):
                tx_batch_norm = CSH.CategoricalHyperparameter('transformer_batch_norm', encoder_params['transformer_batch_norm'])
                tx_num_layers = CSH.CategoricalHyperparameter('transformer_num_layers', encoder_params['transformer_num_layers'])
                tx_embedding_dim = CSH.CategoricalHyperparameter('transformer_embedding_dim', encoder_params['transformer_embedding_dim'])
                tx_n_head = CSH.CategoricalHyperparameter('transformer_n_head', encoder_params['transformer_n_head'])
                tx_ff_num_layers = CSH.CategoricalHyperparameter('transformer_ff_num_layers', encoder_params['transformer_ff_num_layers'])
                tx_max_length = CSH.CategoricalHyperparameter('max_seq_length', encoder_params['max_seq_length'])
                tx_pos_encoding = CSH.CategoricalHyperparameter('positional_encoding_type', encoder_params['positional_encoding_type'])
                cs.add_hyperparameters([tx_batch_norm, tx_num_layers,tx_embedding_dim,tx_n_head,tx_ff_num_layers,tx_max_length,tx_pos_encoding ])

        return cs

