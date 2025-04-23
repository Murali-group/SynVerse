from models.runner import *
from network_rewire import *
from feature_shuffle import shuffle_features
from plots.plot_utils import *
from utils import *

class BaseRunManager:
    def __init__(self, params, model_info, given_epochs, train_df, train_idx, val_idx,
                 dfeat_dict, cfeat_dict, test_df, drug_2_idx,cell_line_2_idx,out_file_prefix,file_prefix, device, **kwargs):
        self.params = params
        self.model_info = model_info
        self.given_epochs = given_epochs
        self.train_df = train_df
        self.train_idx = train_idx
        self.val_idx = val_idx
        self.dfeat_dict = dfeat_dict
        self.cfeat_dict = cfeat_dict
        self.test_df = test_df
        self.drug_2_idx = drug_2_idx
        self.cell_line_2_idx = cell_line_2_idx
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

        hyperparam_file = self.out_file_prefix + '_best_hyperparam.txt'
        if os.path.exists(hyperparam_file):
            hyperparam, _ = extract_best_hyperparam(hyperparam_file)
        else:
            print(f'File: {hyperparam_file} not found for best hyperparam.')
            print('Running with default hyperparameters.')


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
                rewired_train_file = f'{split_file_path}{rand_net}all_train_rewired_{rewire_method}.tsv'
                rewired_df, rewired_train_idx, rewired_val_idx = get_rewired_train_val(
                    self.train_df, self.params.score_name, rewire_method,
                    self.params.split['type'], self.params.split['val_frac'], seed=None,
                    rewired_train_file=rewired_train_file, force_run=False)


                # plot degree and strength distribution of nodes in rewired vs. orig networks.
                #Uncomment the following when want to plot
                # wrapper_network_rewiring_box_plot(rewired_df, self.train_df, self.params.score_name, self.cell_line_2_idx, weighted=True,
                #                                   plot_file_prefix =f'{split_file_path}{rand_net}_{rewire_method}')
                # wrapper_network_rewiring_joint_plot(rewired_df, self.train_df, self.params.score_name, self.cell_line_2_idx, weighted=True, plot_file_prefix=f'{split_file_path}{rand_net}_{rewire_method}')
                #
                # wrapper_network_rewiring_box_plot(rewired_df, self.train_df, self.params.score_name,
                #                                   self.cell_line_2_idx, weighted=False,
                #                                   plot_file_prefix=f'{split_file_path}{rand_net}_{rewire_method}')
                # wrapper_network_rewiring_joint_plot(rewired_df, self.train_df, self.params.score_name,
                #                                     self.cell_line_2_idx, weighted=False,
                #                                     plot_file_prefix=f'{split_file_path}{rand_net}_{rewire_method}')
                #
                out_file_prefix_rewire = f'{self.out_file_prefix}_rewired_{rand_net}_{rewire_method}'
                self.execute_run(rewired_df, rewired_train_idx, rewired_val_idx, self.dfeat_dict, self.cfeat_dict, out_file_prefix_rewire)

class RandomizeScoreRunManager(BaseRunManager):
    def run_wrapper(self):
        for rand_version in range(10):
            # randomized_train_file = f'{split_file_path}{rand_version}all_train_randomized_score.tsv'
            randomized_df = copy.deepcopy(self.train_df)
            randomized_df[self.params.score_name] = randomized_df[self.params.score_name].sample(frac=1).reset_index(drop=True)
            out_file_prefix_randomized = f'{self.out_file_prefix}_randomized_score_{rand_version}'
            self.execute_run(randomized_df,  self.train_idx, self.val_idx, self.dfeat_dict, self.cfeat_dict, out_file_prefix_randomized)

class RunManagerFactory:
    @staticmethod
    def get_run_manager(params, model_info, given_epochs, train_df, train_idx, val_idx,
                        dfeat_dict, cfeat_dict, test_df, drug_2_idx, cell_line_2_idx,
                        out_file_prefix, file_prefix, device, **kwargs):

        train_type = kwargs.get('train_type')
        cls = {
            "regular": BaseRunManager,
            "rewire": RewireRunManager,
            "shuffle": ShuffleRunManager,
            "randomized_score": RandomizeScoreRunManager,
        }.get(train_type, BaseRunManager)

        return cls(params, model_info, given_epochs, train_df, train_idx, val_idx,
                   dfeat_dict, cfeat_dict, test_df, drug_2_idx, cell_line_2_idx,
                   out_file_prefix, file_prefix, device, **kwargs)





