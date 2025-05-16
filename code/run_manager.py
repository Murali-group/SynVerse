import os.path

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
            print('Found hyperparam file: ', out_file_prefix)
        else:
            if self.kwargs.get('use_best_hyperparam'): #user asked to use best hyperparam but the file saving the params are not found. So exit.
                sys.exit(f"Error: hyperparameter file not found ({hyperparam_file}) but use_best_hyperparam=True")
            else:
                print(f'File: {hyperparam_file} not found for best hyperparam. Running with default hyperparameters.')

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

            del shuffled_dfeat_dict
            del shuffled_cfeat_dict

class RewireRunManager(BaseRunManager):
    def run_wrapper(self):
        split_file_path = self.kwargs.get('split_file_path')
        for rewire_method in self.params.rewire_method:
            for rand_net in range(10):
                out_file_prefix_rewire = f'{self.out_file_prefix}_rewired_{rand_net}_{rewire_method}'

                rewired_train_file = f'{split_file_path}{rand_net}all_train_rewired_{rewire_method}.tsv'
                rewired_df, rewired_train_idx, rewired_val_idx = get_rewired_train_val(
                    self.train_df, self.params.score_name, rewire_method,
                    self.params.split['type'], self.params.split['val_frac'], seed=None,
                    rewired_train_file=rewired_train_file, force_run=False)


                # plot degree and strength distribution of nodes in rewired vs. orig networks.
                #Uncomment the following when want to plot
                wrapper_network_rewiring_box_plot(rewired_df, self.train_df, self.params.score_name, self.cell_line_2_idx, weighted=True,
                                                  plot_file_prefix =f'{split_file_path}{rand_net}_{rewire_method}')
                wrapper_network_rewiring_box_plot(rewired_df, self.train_df, self.params.score_name,
                                                  self.cell_line_2_idx, weighted=False,
                                                  plot_file_prefix=f'{split_file_path}{rand_net}_{rewire_method}')
                self.execute_run(rewired_df, rewired_train_idx, rewired_val_idx, self.dfeat_dict, self.cfeat_dict, out_file_prefix_rewire)

class RandomizeScoreRunManager(BaseRunManager):
    def run_wrapper(self):
        for rand_version in range(5):
            # randomized_train_file = f'{split_file_path}{rand_version}all_train_randomized_score.tsv'
            randomized_df = pd.DataFrame()

            edge_types = set(self.train_df['edge_type'].unique())
            for edge_type in edge_types:
                edge_pos_df = self.train_df[(self.train_df['edge_type'] == edge_type) & (self.train_df[self.params.score_name]>=0)]
                edge_pos_df[self.params.score_name] = edge_pos_df[self.params.score_name].sample(frac=1).values
                edge_neg_df = self.train_df[(self.train_df['edge_type'] == edge_type) & (self.train_df[self.params.score_name]<0)]
                edge_neg_df[self.params.score_name] = edge_neg_df[self.params.score_name].sample(frac=1).values
                randomized_df = pd.concat([randomized_df, edge_pos_df, edge_neg_df], axis=0)

            def compute_deviation_of_score(all_train_df, rewired_train_df, score_name):
                # find deviation of score among the overlapping triplets between original and rewired network
                merged = all_train_df[['source', 'target', 'edge_type', score_name]] \
                    .merge(
                    rewired_train_df[['source', 'target', 'edge_type', score_name]],
                    on=['source', 'target', 'edge_type'],
                    suffixes=('_orig', '_rewired')
                )
                # 2. Compute the difference (orig minus rewired)
                merged['score_diff'] = abs(merged[f'{score_name}_orig'] - merged[f'{score_name}_rewired'])
                print(f'average difference btn the same triplet present in original and newired network: ',
                      merged['score_diff'].mean())
                return merged

            deviation = compute_deviation_of_score(randomized_df, self.train_df, self.params.score_name)

            # randomized_df[self.params.score_name] = randomized_df[self.params.score_name].sample(frac=1).reset_index(drop=True)
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





