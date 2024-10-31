from models.model_utils import *
import torch
import pytz
import datetime
from abc import ABC, abstractmethod
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

import hpbandster.core.nameserver as hpns
from hpbandster.optimizers import BOHB as BOHB
import hpbandster.core.result as hpres
import wandb

import pickle
import os
import time
import copy
from torch.utils.data import DataLoader, Subset

import logging

class Runner(ABC):
    def __init__(self, train_val_triplets_df, train_idx, val_idx, dfeat_dict,
                 cfeat_dict, score_name, out_file_prefix,
                 params, model_info, device, **kwargs):

        out_file = out_file_prefix + '.txt'
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        self.split_type = kwargs.get('split_type')
        self.triplets_scores_dataset = self.get_triplets_score_dataset(train_val_triplets_df, score_name=score_name)

        self.drug_feat = dfeat_dict['value']
        self.cell_line_feat = cfeat_dict['value']
        self.dfeat_dim_dict = dfeat_dict['dim']
        self.cfeat_dim_dict = cfeat_dict['dim']

        self.train_idx = train_idx
        self.val_idx = val_idx
        self.n_folds = len(val_idx.keys())
        self.out_file = out_file
        self.out_file_prefix = out_file_prefix
        self.wandb = params.wandb
        self.is_wandb = self.wandb.enabled
        self.bohb_params = params.bohb
        self.device = device
        self.params=params
        self.model_info = model_info

        self.check_freq = 2 #previously ran with 2 , 3
        self.tolerance = 45 #previously ran with 15, 30
        self.batch_size = int(params.batch_size)

        self.result_logger = hpres.json_result_logger(directory=out_file.replace('.txt',''), overwrite=True)
        self.log_file = self.out_file_prefix + '_training.log'
        open(self.log_file,'w')

    @abstractmethod
    def init_model(self, config):
       pass

    @staticmethod
    def get_triplets_score_dataset(triplets_df, score_name='S_mean_mean'):
        triplets = torch.tensor(triplets_df[['source', 'target', 'edge_type']].values)
        synergy_scores = torch.tensor(triplets_df[score_name].values)
        triplets_scores_dataset = TensorDataset(triplets, synergy_scores)
        return triplets_scores_dataset


    def find_best_hyperparam(self, server_type, **kwargs):
        min_budget = self.bohb_params['min_budget']
        max_budget = self.bohb_params['max_budget']
        n_iterations = self.bohb_params['n_iterations']

        if server_type == 'local':
            # get the used specified setting here about wandb and BOHB
            run_id = self.bohb_params['run_id']
            name_server = '127.0.0.1'

            # Step 1: Start a nameserver
            NS = hpns.NameServer(run_id=run_id, host=name_server, port=None)
            NS.start()

            # Step 2: Start a worker #Nure: Model specific
            w = self.worker_cls(self, sleep_interval=0, nameserver=name_server, run_id=run_id)
            w.run(background=True)

            # Step 3: Run an optimizer
            # The run method will return the `Result` that contains all runs performed.

            bohb = BOHB(configspace=w.get_configspace(self.model_info),
                        run_id=run_id, nameserver=name_server,
                        result_logger=self.result_logger,
                        min_budget=min_budget, max_budget=max_budget)
            res = bohb.run(n_iterations=n_iterations)

        elif server_type == 'cluster':
            n_workers = kwargs.get('n_workers')
            worker = kwargs.get('worker')
            run_id = kwargs.get('run_id')
            nic_name = kwargs.get('nic_name')
            shared_directory = kwargs.get('shared_directory')

            # Every process has to lookup the hostname
            host = hpns.nic_name_to_host(nic_name)

            if worker:
                time.sleep(5)  # short artificial delay to make sure the nameserver is already running
                w = self.worker_cls(self, sleep_interval=0.5, run_id=run_id, host=host) #Nure: Model specific
                w.load_nameserver_credentials(working_directory=shared_directory)
                w.run(background=False)
                exit(0)

            # Start a nameserver:
            # We now start the nameserver with the host name from above and a random open port (by setting the port to 0)
            NS = hpns.NameServer(run_id=run_id, host=host, port=0, working_directory=shared_directory)
            ns_host, ns_port = NS.start()

            # Most optimizers are so computationally inexpensive that we can affort to run a
            # worker in parallel to it. Note that this one has to run in the background to
            # not plock!
            #TODO: make sure worker_cls is functioning properly.
            w = self.worker_cls(self, sleep_interval=0.5, run_id=run_id, host=host, nameserver=ns_host,
                          nameserver_port=ns_port)
            w.run(background=True)

            # Run an optimizer
            # We now have to specify the host, and the nameserver information
            bohb = BOHB(configspace=self.worker_cls.get_configspace(self.model_info),
                        run_id=run_id,
                        host=host,
                        nameserver=ns_host,
                        nameserver_port=ns_port,
                        min_budget=min_budget, max_budget=max_budget
                        )
            res = bohb.run(n_iterations=n_iterations, min_n_workers=n_workers)

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

        # extract the average number of epochs the best model(using early stopping) with max_budget
        # has been trained on
        best_epochs = res.data[incumbent].results[max_budget]['n_epochs']

        print('Best found configuration:', id2config[incumbent]['config'])
        print('A total of %i unique configurations where sampled.' % len(id2config.keys()))
        print('A total of %i runs where executed.' % len(res.get_all_runs()))
        print('Total budget corresponds to %.1f full function evaluations.' % (
                sum([r.budget for r in res.get_all_runs()]) / max_budget))

        # save all the result
        bohb_result_file = self.out_file.replace('.txt', '_result.pkl')
        with open(bohb_result_file, 'wb') as fh:
            pickle.dump(res, fh)

        return best_config, best_epochs


    def train_model_given_config(self, config, best_n_epochs, validation = False):

        if not validation: #train model with both training and validation data
            # load dataset
            model, optimizer, criterion = self.init_model(config)
            train_loader = DataLoader(self.triplets_scores_dataset, batch_size=self.batch_size, shuffle=True)
            # train model using the whole training data (including validation dataset)
            best_model_state,_,train_loss, _ = self.train_model(model, optimizer, criterion, train_loader,
                                                      best_n_epochs, self.check_freq,
                                                      self.tolerance, self.is_wandb, self.device, early_stop=False)
            # save the best model
            model_file = self.out_file.replace('.txt', '_model.pth')
            torch.save(best_model_state, model_file)

            #save train_loss
            loss_file = self.out_file.replace('.txt', '_train_loss.txt')
            with open(loss_file, 'w') as file:
                file.write(f'Best config: {config}\n\n')
                file.write(f'Number of epochs: {best_n_epochs}\n\n')
                file.write(f'train_loss: {train_loss}\n\n')
        else:
            val_loss = {}
            train_loss = {}
            req_epochs = {}
            loss_file = self.out_file.replace('.txt', '_train_val_loss.txt')
            file = open(loss_file, 'w')
            file.write(f'Config: {config}\n\n')

            for fold in range(self.n_folds):
                model, optimizer, criterion = self.init_model(config)
                print('FOLD: ', fold)
                fold_train_idx = self.train_idx[fold]
                fold_val_idx = self.val_idx[fold]
                train_subsampler = Subset(self.triplets_scores_dataset, fold_train_idx)
                val_subsampler = Subset(self.triplets_scores_dataset, fold_val_idx)

                train_loader = DataLoader(train_subsampler, batch_size=self.batch_size, shuffle=True)
                val_loader = DataLoader(val_subsampler, batch_size=self.batch_size, shuffle=False)

                best_model_state, val_loss[fold], train_loss[fold], req_epochs[fold] = self.train_model(model, optimizer,
                                    criterion, train_loader, best_n_epochs, self.check_freq,self.tolerance,
                                    self.is_wandb, self.device,early_stop=True,val_loader=val_loader, fold=fold)

                file.write(f'fold: {fold}\n')
                file.write(f'Number of epochs: {req_epochs[fold]}\n')
                file.write(f'train_loss: {train_loss[fold]}\n')
                file.write(f'val_loss: {val_loss[fold]}\n\n')

        return best_model_state, train_loss

    def _init_wandb(self, model, fold):
        wandb.login(key=self.wandb.token)

        # Generate a dynamic run name
        eastern = pytz.timezone(self.wandb.timezone)
        run_name = f"run-{self.split_type}-{fold}-{datetime.datetime.now(eastern).strftime(self.wandb.timezone_format)}"
        wandb.init(project=self.wandb.project_name, entity=self.wandb.entity_name, name=run_name)
        wandb.watch(model, log="all")

    def train_model(self, model, optimizer, criterion, train_loader, n_epochs, check_freq, tolerance, is_wandb, device,
                    early_stop=True, val_loader=None, fold=-1):

        f = open(self.log_file, 'a')
        f.write(f'Configuraion: {model.chosen_config}\n')
        f.write(f"drug_encoder_list: {self.drug_encoder_info}")
        f.write(f"cell_encoder_list: {self.cell_encoder_info}")

        print('Model training starts')
        # if (is_wandb) & (n_epochs>200):  # plot loss with wandb
        if (is_wandb):  # plot loss with wandb
            self._init_wandb(model, fold)

        min_val_loss = 1000000
        req_epochs = n_epochs

        model.train()
        idle_epochs = 0

        # Scheduler to reduce learning rate on plateau
        # TODO: remove after effect of not using scheduler is investigated
        # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

        for i in range(int(n_epochs)):
            train_loss = 0
            for inputs, targets in train_loader:
                t1 = time.time()
                #add (d2,d1,c) triplets for each corresponding (d1,d2,c) triplets.
                inputs_og = copy.deepcopy(inputs)
                inputs[:, [0,1]] = inputs[:, [1,0]]
                inputs_undir = torch.cat((inputs_og, inputs), dim=0)
                targets_undir = torch.cat((targets, targets), dim=0).to(device)

                optimizer.zero_grad()
                outputs = model(inputs_undir, self.drug_feat, self.cell_line_feat, device)
                loss = criterion(outputs, targets_undir.reshape(-1, 1))
                train_loss += (loss.detach().cpu().numpy())
                loss.backward()
                optimizer.step()

            train_loss = train_loss / len(train_loader)
            print('e: ', i, '  train_loss: ', train_loss)
            f.write(f'e {i}: train_loss: {train_loss}\n')

            if (val_loader is not None):
                if (i % check_freq) == 0:
                    # checkpoint to provide early stop mechanism
                    val_loss = self.eval_model(model, val_loader, criterion, device)
                    model.train()

                    #TODO: remove after effect of not using scheduler is investigated
                    # Step the scheduler with the validation loss
                    # scheduler.step(val_loss)

                    print('                                   e: ', i, '  val_loss: ', val_loss)
                    f.write(f'\n------------------------------e {i}: val_loss: {val_loss}\n\n')

                    if (is_wandb):  # plot loss with wandb
                        wandb.log({"epoch": i, "train_loss": train_loss, "val_loss": val_loss})

                    if early_stop:
                        # if for number of tolerance epochs validation loss did not decrease then return last best model.
                        idle_epochs += 1
                        if val_loss < min_val_loss:
                            min_val_loss = val_loss
                            idle_epochs = 0  # reset idle epochs
                            best_model = model.state_dict()
                            # torch.save(best_model, 'best_MLP_model.pth')
                        if idle_epochs > tolerance:
                            req_epochs = i
                            # if (is_wandb) & (n_epochs > 200):
                            if is_wandb:
                                wandb.finish()
                            f.close()
                            return best_model, min_val_loss, train_loss, req_epochs


            else:
                if (is_wandb) and ((i % check_freq) == 0):
                    wandb.log({"epoch": i, "train_loss": train_loss})

        if not early_stop:
            best_model = model.state_dict()

        if is_wandb:
            wandb.finish()
        f.close()
        return best_model, min_val_loss, train_loss, req_epochs  # model has been trained for given number of epochs. Now return the best model so far.


    def eval_model(self, model, val_loader, criterion, device, save_output=False):
        '''
        Return validation loss per sample.
        '''
        # Evaluate after training
        model.eval()

        with torch.no_grad():
            if save_output:
                triplets = []
                true_score = []
                pred_score = []
            total_loss = 0
            for inputs, targets in val_loader:
                inputs_og = copy.deepcopy(inputs)
                inputs[:, [0, 1]] = inputs[:, [1, 0]] #make input undirected,i.e., bot (a,b) and (b,a) drug pairs are present.
                inputs_undir = torch.cat((inputs_og, inputs), dim=0)
                targets_undir = torch.cat((targets, targets), dim=0).to(device)
                outputs = model(inputs_undir, self.drug_feat, self.cell_line_feat, device)
                loss = criterion(outputs, targets_undir.reshape(-1, 1))
                total_loss += (loss.detach().cpu().numpy())

                if save_output:
                    triplets.append(inputs_undir.to('cpu'))
                    true_score.append(targets_undir.reshape(-1, 1).to('cpu'))
                    pred_score.append(outputs.to('cpu'))

        # loss_per_sample
        avg_loss = total_loss / len(val_loader)
        if save_output:#save predicted scores
            out_file = self.out_file.replace('.txt', '_test_predicted_scores.tsv')

            triplets = torch.cat(triplets, dim=0).numpy()
            true_score = torch.cat(true_score, dim=0).numpy()
            pred_score = torch.cat(pred_score, dim=0).numpy()
            # Create a DataFrame
            df = pd.DataFrame(triplets, columns=['drug1', 'drug2', 'cell_line'])
            df['true'] = true_score
            df['predicted'] = pred_score
            # Save the DataFrame to a tab-separated file
            df.to_csv(out_file, sep='\t', index=False)
            print(f"Predicted score saved to {out_file}")

        return avg_loss

    def get_test_score(self, test_df, best_model_state, config, best_n_epochs):
        test_loader = DataLoader(self.get_triplets_score_dataset(test_df), batch_size=4096, shuffle=True)
        # evaluate model on test dataset
        model, optimizer, criterion = self.init_model(config)
        model.load_state_dict(best_model_state)
        test_loss = self.eval_model(model, test_loader, criterion, self.device, save_output=True)
        print('test loss: ', test_loss)
        # save test loss result

        out_file =self.out_file.replace('.txt', '_test_loss.txt')
        with open(out_file, 'w') as file:
            file.write(f'Best config: {config}\n\n')
            file.write(f'Number of epochs: {best_n_epochs}\n\n')
            file.write(f'test_loss: {test_loss}\n\n')

        file.close()
