import copy
import os.path
import pandas as pd
from evaluation.split_generalized import *
from utils import *
from plot_utils import plot_dist
import types
import argparse
from models.encoder_mlp_runner import *
from cell_line_preprocess import *
from drug_preprocess import *
from network_algorithms.rwr_runner import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    group.add_argument('--config', type=str, default="/home/grads/tasnina/Projects/SynVerse/code/"
                       "config_files/experiment_1/d1hot_c1hot.yaml",
                       help="Configuration file for this script.")

    group.add_argument('--feat', type=str,
                       help="Put the name of the features to use, separated by space.")
    group.add_argument('--split', type=str,
                       help="Put the name of the split types to run, separated by space.")
    group.add_argument('--n_workers', type=int, help='Number of workers to run in parallel.', default=2)
    group.add_argument('--worker', help='Flag to turn this into a worker process', action='store_true')
    group.add_argument('--run_id', type=str,
                        help='A unique run id for this optimization run. An easy option is to use the job id of the clusters scheduler.')
    group.add_argument('--nic_name', type=str, default = 'eno1', help='Which network interface to use for communication.'
                        'The valid interface names for VT arc is among: [lo, eno1, enp33s0f0, eno2, enp33s0f1, ib0]')
    group.add_argument('--shared_directory', type=str,
                        help='A directory that is accessible for all processes, e.g. a NFS share.')

    return parser


def run_SynVerse(inputs, params, **kwargs):
    #TODO: set default values for the params if not given in config file.
    print(device)
    print('SYNVERSE STARTING')
    drug_features = params.drug_features
    cell_line_features = params.cell_line_features
    model_info = params.models
    splits = params.splits
    split_dir = params.split_dir
    synergy_file = inputs.processed_syn_file
    score_name = 'S_mean_mean' #synergy score to use


    '''Read synergy triplets'''
    synergy_df = pd.read_csv(synergy_file, sep='\t', dtype={'drug_1_pid': str, 'drug_2_pid': str, 'cell_line_name': str})
    drug_pids = sorted(list(set(synergy_df['drug_1_pid']).union(set(synergy_df['drug_2_pid']))))
    cell_line_names = sorted(synergy_df['cell_line_name'].unique())


   #********************************** GET FEATURES READY *******************************************************
    ''' Read parsed drug features and do user-chosen filtering and preprocessing.'''
    dfeat_dict, dfeat_names = prepare_drug_features(drug_features, drug_pids, params, inputs)

    ''' Read parsed cell line features and do user-chosen filtering and preprocessing.'''
    cfeat_dict, cfeat_names = prepare_cell_line_features(cell_line_features, cell_line_names, params, inputs)

    '''Filter out the triplets based on the availability of drug and cell line features'''
    synergy_df = feature_based_filtering(synergy_df, dfeat_dict['value'], cfeat_dict['value'], params.feature)

    '''keep the cell lines consisting of at least params.abundance% of the total #triplets in the final dataset.'''
    synergy_df = abundance_based_filtering(synergy_df, min_frac=params.abundance)

    #******************************************* MODEL TRAINING ***********************************************

    #***********************************************Figure out the feature combinations to train the model on ***
    #if I put --feat as user defined parameter, then I want to use only that combination of feature. override whatever is
    #given in the config
    use_feat = kwargs.get('feat')
    drug_cell_feat_combs = get_feature_comb_wrapper(dfeat_names, dfeat_dict, cfeat_names, cfeat_dict,
                             use_feat=use_feat, max_feat=params.max_feat)


    ''' prepare split'''
    for split in splits:
        split_type = split['type']
        n_folds = split['n_folds']
        test_frac = split['test_frac']
        print('SPLIT: ', split_type)

        #if user defined split type is present as kwargs param, then only the split types common between config and kwargs param
        #will run.
        udef_split_types = kwargs.get('split')
        if udef_split_types is not None:
            udef_split_types = udef_split_types.split(' ')
            if split_type not in udef_split_types: #do not run split type not present in kwargs param
                continue

        #split into train test
        split_feat_str = get_feat_prefix(dfeat_dict, cfeat_dict)
        split_prefix = split_dir + f'/{split_feat_str}/k_{params.abundance}/{split_type}_{test_frac}_{n_folds}/'

        force_split = False

        train_df, test_df, drug_2_idx, cell_line_2_idx = wrapper_train_test(copy.deepcopy(synergy_df), split_type, test_frac, split_prefix, force_run=force_split)
        #plot synergy score distribution for train and test set
        plot_dist(train_df[score_name], 'train', out_dir=split_prefix)
        plot_dist(test_df[score_name], 'test', out_dir=split_prefix)


        #split into train_val for n_folds
        train_idx, val_idx = wrapper_nfold_split(train_df, split_type, n_folds, split_prefix, force_run=force_split)

        #convert feature dataframes into numpy arrays while in the array row i corresponds to the drug with numerical idx i
        cur_dfeat_dict = copy.deepcopy(dfeat_dict)
        cur_cfeat_dict = copy.deepcopy(cfeat_dict)
        #TODO make sure that tokenized smiles is an array.
        cur_dfeat_dict['value'], cur_cfeat_dict['value'] = get_index_sorted_feature_matrix(cur_dfeat_dict['value'], drug_2_idx,
                                               cur_cfeat_dict['value'], cell_line_2_idx)

        #Normalize data based on training data. Use the mean, std from training data to normalize test data.
        cur_dfeat_dict['value'], cur_cfeat_dict['value'] = normalization_wrapper(cur_dfeat_dict['value'], cur_cfeat_dict['value'],
                                                                    cur_dfeat_dict['norm'], cur_cfeat_dict['norm'], train_df)

        for (select_drug_feat, select_cell_feat) in drug_cell_feat_combs:
            print('drug and cell line features in use: ', select_drug_feat, select_cell_feat)

            # only keep the selected drug and cell feature for training and further analysis
            select_dfeat_dict = keep_selected_feat(cur_dfeat_dict, select_drug_feat)
            select_cfeat_dict = keep_selected_feat(cur_cfeat_dict, select_cell_feat)
            # depending on the selected encoders modify the model architecture here.
            select_model_info = get_select_model_info(model_info, select_dfeat_dict['encoder'],
                                                      select_cfeat_dict['encoder'])

            hyperparam = combine_hyperparams(select_model_info)
            best_n_epochs = params.epochs

            out_file_prefix = create_file_prefix(params, select_dfeat_dict, select_cfeat_dict, split_type, split_feat_str=split_feat_str)

            # out_file_prefix = params.out_dir+'/test.txt'
            kwargs['split_type'] = split_type
            runner = Encode_MLP_runner(train_df, train_idx, val_idx, select_dfeat_dict, select_cfeat_dict, score_name,
                     out_file_prefix, params, select_model_info, device, **kwargs)

            if params.mode == 'hp_tune':
                # find best hyperparam setup
                hyperparam, best_n_epochs = runner.find_best_hyperparam(params.bohb['server_type'], **kwargs)
                # train the model with best hyperparam and both train and validation dataset
                trained_model_state, train_loss = runner.train_model_given_config(hyperparam, best_n_epochs)
                # evaluate model on test data
                test_loss = runner.get_test_score(test_df, trained_model_state, hyperparam, best_n_epochs)

            elif params.mode== 'train_val':
                trained_model_state, train_loss = runner.train_model_given_config(hyperparam, best_n_epochs,
                                                                                  validation=True)
            elif params.mode == 'train':
                # train the model with best hyperparam and both train and validation dataset
                trained_model_state, train_loss = runner.train_model_given_config(hyperparam, best_n_epochs)
                # evaluate model on test data
                test_loss = runner.get_test_score(test_df, trained_model_state, hyperparam, best_n_epochs)


        del cur_dfeat_dict
        del cur_cfeat_dict
def main(config_map, **kwargs):

    if 'snakemake' in globals():
        # drug_features = snakemake.params[0],
        # cell_line_features = snakemake.params[1],
        # models = snakemake.params[2],
        run_SynVerse(snakemake.input, snakemake.params, **kwargs)
    else:
        # config_map = load_yaml_file(config_map)
        input_dir = config_map['input_settings']['input_dir']
        output_dir = config_map['output_settings']['output_dir']

        inputs = types.SimpleNamespace()
        params = types.SimpleNamespace()

        inputs.processed_syn_file = input_dir + 'synergy/synergy_scores.tsv'
        inputs.drug_smiles_file = input_dir + 'drug/smiles.tsv'
        inputs.drug_graph_file = input_dir + 'drug/molecular_graph.pickle'
        inputs.drug_target_file = input_dir + 'drug/target.tsv'
        # inputs.vocab = input_dir + 'drug/vocab_bpe_300.txt'
        # inputs.spmm_checkpoint = input_dir + 'drug/pretrain/checkpoint_SPMM.ckpt'


        inputs.cell_line_file = input_dir + 'cell-line/gene_expression.tsv'
        inputs.lincs = input_dir + 'cell-line/LINCS_1000.txt'

        params.drug_features = config_map['input_settings']['drug_features']
        params.cell_line_features = config_map['input_settings']['cell_line_features']
        params.models = config_map['input_settings']['models']
        params.epochs = config_map['input_settings']['epochs']

        params.splits = config_map['input_settings']['splits']
        params.feature = config_map['input_settings']['feature']
        params.abundance = config_map['input_settings']['abundance']
        params.max_feat=config_map['input_settings']['max_feat']
        params.mode=config_map['input_settings']['mode']
        params.batch_size = config_map['input_settings'].get('batch_size', 4096)
        input_settings = config_map.get('input_settings', {})
        params.wandb = types.SimpleNamespace(**input_settings.get('wandb', {}))
        params.bohb = config_map['input_settings']['bohb']
        params.drug_chemprop_dir = input_dir + '/drug/chemprop/'
        params.out_dir = output_dir
        params.split_dir = input_dir + 'splits'
        run_SynVerse(inputs, params, **kwargs)


if __name__ == "__main__":
    config_map, kwargs = parse_args()
    main(config_map, **kwargs)




