from preprocessing.preprocess import load_filter_triplets_features, post_split_processing
from split import *
from utils import *
from run_manager import RunManagerFactory
from models.runner import *
from parse_config import parse_config
import argparse
from plots.plot_utils import plot_synergy_data_dist

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = get_available_gpu()
def parse_args():
    parser = setup_opts()
    args = parser.parse_args()
    kwargs = vars(args)
    with open(args.config, 'r') as conf:
        # config_map = yaml.load(conf, Loader=yaml.FullLoader)
        config_map = yaml.safe_load(conf)
    return config_map, kwargs

def setup_opts():
    ## Parse command line args.
    parser = argparse.ArgumentParser(description="""Script to download and parse input files, and (TODO) run the  pipeline using them.""")
    # general parameters
    group = parser.add_argument_group('Main Options')
    group.add_argument('--config', type=str, default="config_files/Transformer_Berttoken.yaml", help="Configuration file for this script.")

    group.add_argument('--train_type', type=str, default="regular",
                       help="Three Options. ['regular','rewire','shuffle','randomized_score]."
                            "'regular => train and test model with original feature and triplets, "
                            "'rewire' => randomize train triplets, 'shuffle' => shuffle features."
                            "'randomized_score' => randomize the score of the triplets. ")

    group.add_argument('--use_best_hyperparam', type=bool, default=True,
                       help="True =>Search for the file where best hyperparam is saved, if not found then exit. False => If file for best hyperparam is present then use that otherwise run with default params.")
    group.add_argument('--seed', type=int, default=0,
                       help="Seed value used for train test splitting. Using different seed value will result in different train and test splits.")
    group.add_argument('--feat', type=str,
                       help="Put the name of the features to use, separated by space. Applicable when you want to run just one set of features.")
    group.add_argument('--split', type=str,
                       help="Put the name of the split types to run, separated by space.")
    group.add_argument('--force_split', type=bool, default=False,
                       help="Should split or use the existing splits.")
    group.add_argument('--start_run', type=int, default=0, help='From which run should the model start from. This is to help when'
                    'some model has been trained for first 2 runs but then terminated by arc. Then next time we need to start from run 2, hence start_run should be 2')
    group.add_argument('--end_run', type=int, default=5, help='How many runs you want. end_run=5 means we will get runs starting at start_run and ending at (end_run-1)')
    group.add_argument('--n_workers', type=int, help='Number of workers to run in parallel.', default=2)
    group.add_argument('--worker', help='Flag to turn this into a worker process', action='store_true')
    group.add_argument('--run_id', type=str, default = 'synverse',
                        help='A unique run id for this optimization run. An easy option is to use the job id of the clusters scheduler.')
    group.add_argument('--nic_name', type=str, default = 'eno1', help='Which network interface to use for communication.'
                        'The valid interface names for VT arc is among: [lo, eno1, enp33s0f0, eno2, enp33s0f1, ib0]')
    group.add_argument('--shared_directory', type=str,
                        help='A directory that is accessible for all processes, e.g. a NFS share.')
    return parser


def run_SynVerse(inputs, params, **kwargs):
    print(f'SYNVERSE STARTING on {device}')

    seed = kwargs.get('seed')
    start_run, end_run = kwargs.get('start_run'), kwargs.get('end_run')

    #load_triplets and features
    '''Read synergy triplets'''
    synergy_df = pd.read_csv(inputs.synergy_file, sep='\t',
                             dtype={'drug_1_pid': str, 'drug_2_pid': str, 'cell_line_name': str,
                                    params.score_name: float})
    drug_pids = sorted(list(set(synergy_df['drug_1_pid']).union(set(synergy_df['drug_2_pid']))))
    cell_line_names = sorted(synergy_df['cell_line_name'].unique())

    synergy_df, dfeat_dict, cfeat_dict, drug_2_idx, cell_line_2_idx = load_filter_triplets_features(synergy_df, drug_pids, cell_line_names, inputs, params, device)

    #get the feature combinations to train the model
    feat_str = get_feat_prefix(dfeat_dict, cfeat_dict)
    drug_cell_feat_combs = get_feature_comb_wrapper(dfeat_dict, cfeat_dict,
                            max_drug_feat=params.max_drug_feat,
                            min_drug_feat = params.min_drug_feat, max_cell_feat=params.max_cell_feat, min_cell_feat = params.min_cell_feat)

    # plot_synergy_data_dist(synergy_df, params.score_name, title = feat_str, out_file = f'{params.input_dir}/synergy/data_distribution_{feat_str}_{params.score_name}.pdf')
    for run_no in range(start_run, end_run):
        for split in params.splits:
            split_type, test_frac, val_frac, params.split = split['type'], split['test_frac'], split['val_frac'], split
            split_info_str = f"/{feat_str}/k_{params.abundance}_{params.score_name}/{split_type}_{test_frac}_{val_frac}/run_{run_no}_{seed}/"
            split_file_path = params.split_dir + split_info_str
            kwargs['split_file_path'] = split_file_path

            #split into train test val
            test_df, all_train_df, train_idx, val_idx = wrapper_test_train_val(copy.deepcopy(synergy_df), split_type, test_frac, val_frac, split_file_path, seed=seed+run_no,
                                                                               force_run=kwargs.get('force_split'))
            all_train_df = all_train_df[['source', 'target','edge_type', params.score_name]]

            #************************** POST split processing of features ******************************************
            cur_dfeat_dict, cur_cfeat_dict = post_split_processing(dfeat_dict, cfeat_dict, all_train_df, params, split_info_str, device)

            for (select_drug_feat, select_cell_feat) in drug_cell_feat_combs:
                print('drug and cell line features in use: ', select_drug_feat, select_cell_feat)

                # only keep the selected drug and cell feature for training and further analysis
                select_dfeat_dict = keep_selected_feat(cur_dfeat_dict, select_drug_feat)
                select_cfeat_dict = keep_selected_feat(cur_cfeat_dict, select_cell_feat)
                # depending on the selected encoders modify the model architecture here.
                select_model_info = get_select_model_info(params.model_info, select_dfeat_dict['encoder'], select_cfeat_dict['encoder'])

                params.hyperparam = combine_hyperparams(select_model_info)
                given_epochs = params.epochs

                out_file_prefix = create_file_prefix(params, select_dfeat_dict, select_cfeat_dict, split_type,
                                                      split_feat_str=feat_str, run_no=run_no, seed=seed)

                #**************************** Run the pipeline to train and test model **********************************************************
                run_manager = RunManagerFactory.get_run_manager(params, select_model_info, given_epochs, all_train_df,
                            train_idx, val_idx, select_dfeat_dict, select_cfeat_dict, test_df, drug_2_idx,cell_line_2_idx, out_file_prefix, '_val_true_', device, **kwargs)
                run_manager.run_wrapper()

            del cur_dfeat_dict, select_drug_feat
            del cur_cfeat_dict, select_cell_feat


if __name__ == "__main__":
    config_map, kwargs = parse_args()
    #parse input_files and params from config_file. If some params are overlapping across config_map and kwargs, then prioritize kwargs.
    inputs, params = parse_config(config_map, **kwargs)
    run_SynVerse(inputs, params, **kwargs)



