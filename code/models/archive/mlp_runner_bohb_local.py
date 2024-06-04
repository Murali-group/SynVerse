from models.mlp_wrapper import *
from models.model_utils import *

import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from models.mlp_worker import MLPWorker
import hpbandster.core.nameserver as hpns
from hpbandster.optimizers import BOHB as BOHB

import os
import pickle

class MLP_runner:
    def __init__(self,train_val_df, train_idx, val_idx, test_df, dfeat_dict, cfeat_dict, drug_2_idx,
                 cell_line_2_idx,out_file_prefix, params, device, **kwargs):

        #feature matrix prep
        #train and val data prep
        node_X = concatenate_features(dfeat_dict, 'pid', drug_2_idx)
        edge_type_X = concatenate_features(cfeat_dict, 'cell_line_name', cell_line_2_idx)
        init_size = len(train_val_df)
        X_train_val = torch.from_numpy(
            prepare_feat_for_MLP(train_val_df[['source', 'target', 'edge_type']], node_X, edge_type_X))
        y_train_val = torch.tensor(list(train_val_df['S_mean_mean'].values) * 2)  # repeat labels

        # Test data prep
        X_test_val = torch.from_numpy(
            prepare_feat_for_MLP(test_df[['source', 'target', 'edge_type']], node_X, edge_type_X))
        y_test_val = torch.tensor(list(test_df['S_mean_mean'].values) * 2)  # repeat labels

        undir_train_idx = {}  # contain both (d1,d2,c1) and (d2,d1,c1), i.e., triplets with undirected drug pairs.
        undir_val_idx = {}

        for fold in train_idx:
            undir_train_idx[fold] = train_idx[fold] + [a + init_size for a in train_idx[fold]]
            undir_val_idx[fold] = val_idx[fold] + [a + init_size for a in val_idx[fold]]

        out_file = out_file_prefix + '.txt'
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        open(out_file, 'w')


        self.train_val_dataset = TensorDataset(X_train_val, y_train_val)
        self.test_dataset = TensorDataset(X_test_val, y_test_val)
        self.undir_train_idx = undir_train_idx
        self.undir_val_idx = undir_val_idx
        self.input_dim = X_train_val.shape[1]
        self.n_folds = len(val_idx.keys())
        self.out_file = out_file
        self.is_wandb = params.wandb
        self.bohb_params = params.bohb
        self.device=device

        self.check_freq=2
        self.tolerance=10

        del (X_train_val, y_train_val, X_test_val, y_test_val)


    def init_model(self, config):
        # initiate and train model
        hidden_layers = []
        for i in range(config['num_hid_layers']):
            hidden_layers.append(config[f'hid_{i}'])

        model = MLP_wrapper(input_size=self.input_dim, hidden_layers=hidden_layers,
                            output_size=1, in_dropout_rate=config['in_dropout_rate'],
                            hid_dropout_rate=config['hid_dropout_rate']).to(self.device)

        ## Wrap the model for parallel processing if multiple gpus are available
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs!")
            model = nn.DataParallel(model)

        criterion = nn.MSELoss()
        if config['optimizer'] == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=config['lr'])
        else:
            optimizer = optim.SGD(model.parameters(), lr=config['lr'], momentum=config['sgd_momentum'])

        return model, optimizer, criterion

    def find_best_hyperparam(self):
    #get the used specified setting here about wandb and BOHB
        run_id = self.bohb_params['run_id']
        min_budget = self.bohb_params['min_budget']
        max_budget = self.bohb_params['max_budget']
        n_iterations = self.bohb_params['n_iterations']
        name_server = '127.0.0.1'


        # Step 1: Start a nameserver
        NS = hpns.NameServer(run_id=run_id, host=name_server, port=None)
        NS.start()

        # Step 2: Start a worker
        w = MLPWorker(self.train_val_dataset, self.undir_train_idx, self.undir_val_idx,
                input_dim=self.input_dim,n_folds=self.n_folds, check_freq=self.check_freq, tolerance=self.tolerance,
                is_wandb=self.is_wandb, device=self.device,
                sleep_interval=0, nameserver = name_server, run_id=run_id)
        w.run(background=True)

        # Step 3: Run an optimizer
        # The run method will return the `Result` that contains all runs performed.

        bohb = BOHB(configspace=w.get_configspace(),
                run_id=run_id, nameserver=name_server,
                min_budget=min_budget, max_budget=max_budget)
        res = bohb.run(n_iterations=n_iterations)

        # Step 4: Shutdown
        # After the optimizer run, we must shutdown the master and the nameserver.
        bohb.shutdown(shutdown_workers=True)
        NS.shutdown()


        # Step 5: Analysis
        # Each optimizer returns a hpbandster.core.result.Result object.
        # It holds informations about the optimization run like the incumbent (=best) configuration.
        # For further details about the Result object, see its documentation.
        # Here we simply print out the best config and some statistics about the performed runs.
        id2config = res.get_id2config_mapping()
        incumbent = res.get_incumbent_id()
        best_config = id2config[incumbent]['config']

        #extract the average number of epochs the best model(using early stopping) with max_budget
        #has been trained on
        best_epochs = res.data[incumbent].results[max_budget]['n_epochs']

        print('Best found configuration:', id2config[incumbent]['config'])
        print('A total of %i unique configurations where sampled.' % len(id2config.keys()))
        print('A total of %i runs where executed.' % len(res.get_all_runs()))
        print('Total budget corresponds to %.1f full function evaluations.' % (
                    sum([r.budget for r in res.get_all_runs()]) / max_budget))


        #save all the result
        bohb_result_file = self.out_file.replace('.txt','_result.pkl')
        with open(bohb_result_file, 'wb') as fh:
            pickle.dump(res, fh)

        return best_config, best_epochs



    def train_model_with_best_hyperparam(self, config, best_n_epochs):
        # load dataset
        train_loader = DataLoader(self.train_val_dataset, batch_size=4096, shuffle=True)

        model, optimizer, criterion = self.init_model(config)
        # train model using the whole training data (including validation dataset)
        #TODO: what epoch number should I use here as I cannot use early stopping anymore.
        best_model_state, _,_ = train_model(model, optimizer, criterion, train_loader,
                                best_n_epochs, self.check_freq,
                                self.tolerance, self.is_wandb, self.device, early_stop=False)
        # save the best model
        model_file = self.out_file.replace('.txt', '_model.pth')
        torch.save(best_model_state, model_file)

        return best_model_state

    def get_test_score(self, best_model_state, config, best_n_epochs):
        test_loader = DataLoader(self.test_dataset, batch_size=4096, shuffle=False)

        #evaluate model on test dataset
        model, optimizer, criterion = self.init_model(config)

        model.load_state_dict(best_model_state)
        test_loss = eval_model(model, test_loader, criterion, self.device)
        print('test loss: ', test_loss)
        #save test loss result
        with open(self.out_file, 'a') as file:
            file.write(f'Best config: {config}\n\n')
            file.write(f'Number of epochs: {best_n_epochs}\n\n')
            file.write(f'test_loss: {test_loss}\n\n')

        file.close()

