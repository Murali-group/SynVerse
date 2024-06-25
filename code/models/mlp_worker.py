try:
    import torch
    import torch.utils.data
    import torch.nn as nn
    import torch.nn.functional as F
except:
    raise ImportError("For this example you need to install pytorch.")
from torch.utils.data import DataLoader, Subset

import ConfigSpace as CS
import ConfigSpace.conditions as CSC
import ConfigSpace.hyperparameters as CSH
from hpbandster.core.worker import Worker
import logging
logging.basicConfig(level=logging.DEBUG)

from models.mlp_wrapper import *
import time

class MLPWorker(Worker):
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

        print('hidden layers: ', config['hid_0'], config['hid_1'], config['hid_2'])
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
        avg_epochs = int(sum(req_epochs.values())/self.runner.n_folds)

        return ({
                'loss': avg_val_loss, # remember: HpBandSter always minimizes!
                'n_epochs': avg_epochs,
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

        pred_params = model_params['hp_range']
        # ****************** GENERAL configurations ***********************************************
        lr = CSH.UniformFloatHyperparameter('lr', lower=pred_params['lr'][0], upper=pred_params['lr'][1],
                                            default_value=1e-4, log=True)
        optimizer = CSH.CategoricalHyperparameter('optimizer', pred_params['optimizer'])
        sgd_momentum = CSH.UniformFloatHyperparameter('sgd_momentum', lower=pred_params.get('sgd_momentum', [0, 0])[0],
                                                      upper=pred_params.get('sgd_momentum', [0.99, 0.99])[1],
                                                      default_value=0.9, log=False)
        cs.add_hyperparameters([lr, optimizer, sgd_momentum])
        # The hyperparameter sgd_momentum will be used,if the configuration
        # contains 'SGD' as optimizer.
        cond = CS.EqualsCondition(sgd_momentum, optimizer, 'SGD')
        cs.add_condition(cond)

        # ********************* Configurations for synergy predicting MLP layer **********************
        num_hid_layers = CSH.UniformIntegerHyperparameter('num_hid_layers', lower=pred_params['num_hid_layers'][0],
                                                          upper=pred_params['num_hid_layers'][1], default_value=2)

        hid_0 = CSH.UniformIntegerHyperparameter('hid_0', lower=pred_params.get('hid_0', [64, 64])[0],
                                                 upper=pred_params.get('hid_0', [2048, 2048])[1], default_value=1024,
                                                 log=True)
        hid_1 = CSH.UniformIntegerHyperparameter('hid_1', lower=pred_params.get('hid_1', [64, 64])[0],
                                                 upper=pred_params.get('hid_1', [2048, 2048])[1], default_value=1024,
                                                 log=True)
        hid_2 = CSH.UniformIntegerHyperparameter('hid_2', lower=pred_params.get('hid_2', [64, 64])[0],
                                                 upper=pred_params.get('hid_2', [2048, 2048])[1], default_value=1024,
                                                 log=True)
        cs.add_hyperparameters([num_hid_layers, hid_0, hid_1, hid_2])
        # add conditions so that hid_2 will be considered when num_hid_layers>1 and so on.
        cond = CS.GreaterThanCondition(hid_1, num_hid_layers, 1)
        cs.add_condition(cond)
        cond = CS.GreaterThanCondition(hid_2, num_hid_layers, 2)
        cs.add_condition(cond)

        # # Define a condition to ensure hid_0 > hid_1
        # cond = CSC.LessThanCondition(hid_0, hid_1, value=hid_1)
        # # Add the condition to the configuration space
        # cs.add_condition(cond)
        #
        # # Define a condition to ensure hid_1 > hid_2
        # cond = CSC.LessThanCondition(hid_1, hid_2, value=hid_2)
        # # Add the condition to the configuration space
        # cs.add_condition(cond)

        in_dropout_rate = CSH.UniformFloatHyperparameter('in_dropout_rate', lower=pred_params['in_dropout_rate'][0],
                                                         upper=pred_params['in_dropout_rate'][1], default_value=0.5,
                                                         log=False)
        hid_dropout_rate = CSH.UniformFloatHyperparameter('hid_dropout_rate', lower=pred_params['hid_dropout_rate'][0],
                                                          upper=pred_params['hid_dropout_rate'][1], default_value=0.5,
                                                          log=False)

        cs.add_hyperparameters([in_dropout_rate, hid_dropout_rate])

        return cs

