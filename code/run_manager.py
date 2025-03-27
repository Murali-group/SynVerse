from models.runner import *
from network_rewire import *
from feature_shuffle import shuffle_features
from utils import *

class BaseRunManager:
    def __init__(self, params, model_info, given_epochs, train_df, train_idx, val_idx,
                 dfeat_dict, cfeat_dict, test_df, out_file_prefix,file_prefix, device, **kwargs):
        self.params = params
        self.model_info = model_info
        self.given_epochs = given_epochs
        self.train_df = train_df
        self.train_idx = train_idx
        self.val_idx = val_idx
        self.dfeat_dict = dfeat_dict
        self.cfeat_dict = cfeat_dict
        self.test_df = test_df
        self.file_prefix = file_prefix
        self.out_file_prefix = out_file_prefix
        self.device = device
        self.kwargs = kwargs

    def execute_run(self, train_df, train_idx, val_idx, dfeat_dict, cfeat_dict, out_file_prefix):

        #pipeline for training a model
        hyperparam = self.params.hyperparam
        runner = Runner(train_df, train_idx, val_idx, dfeat_dict, cfeat_dict,
                        out_file_prefix, self.params, self.model_info,
                        self.device, **self.kwargs)

        if self.params.hp_tune:
            runner.find_best_hyperparam(self.params.bohb['server_type'], **self.kwargs)

        if self.params.train_mode['use_best_hyperparam']:
            hyperparam_file = self.out_file_prefix + '_best_hyperparam.txt'
            hyperparam, _ = extract_best_hyperparam(hyperparam_file)

        trained_model_state, train_loss = runner.train_model_given_config(hyperparam, self.given_epochs, validation=True, save_output=True)

        #testing a model
        runner.get_test_score(self.test_df, trained_model_state, hyperparam, save_output=True, file_prefix=self.file_prefix)

    def run_wrapper(self, *args, **kwargs):
        self.execute_run(self.train_df, self.train_idx, self.val_idx, self.dfeat_dict, self.cfeat_dict, self.out_file_prefix)


class ShuffleRunManager(BaseRunManager):
    def run_wrapper(self):
        for shuffle_no  in range (10):
            # Shuffle features for a single run
            shuffled_dfeat_dict = {**self.dfeat_dict, 'value': shuffle_features(self.dfeat_dict['value'])}
            shuffled_cfeat_dict = {**self.cfeat_dict, 'value': shuffle_features(self.cfeat_dict['value'])}

            out_file_prefix_shuffle = f'{self.out_file_prefix}_shuffled_{shuffle_no}'
            self.execute_run(self.train_df, self.train_idx, self.val_idx, shuffled_dfeat_dict, shuffled_cfeat_dict, out_file_prefix_shuffle)



class RewireRunManager(BaseRunManager):
    def run_wrapper(self):
        split_file_path = self.kwargs.get('split_file_path')
        for rewire_method in self.params.rewire_method:
            for rand_net in range(10):
                out_file_prefix_rewire = f'{self.out_file_prefix}_rewired_{rand_net}_{rewire_method}'
                rewired_df, rewired_train_idx, rewired_val_idx = get_rewired_train_val(
                    self.train_df, self.params.score_name, rewire_method,
                    self.params.split['type'], self.params.split['val_frac'], self.kwargs.get('seed')+rand_net,
                    out_dir=f'{split_file_path}{rand_net}')
                self.execute_run(rewired_df, rewired_train_idx, rewired_val_idx, self.dfeat_dict, self.cfeat_dict, out_file_prefix_rewire)

class RunManagerFactory:
    @staticmethod
    def get_run_manager(train_type, *args, **kwargs):
        if train_type == "rewire":
            return RewireRunManager(*args, **kwargs)
        elif train_type == "shuffle":
            return ShuffleRunManager(*args, **kwargs)
        else:
            return BaseRunManager(*args, **kwargs)


