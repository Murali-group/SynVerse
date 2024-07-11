import os.path

import pandas as pd
from evaluation.split_generalized import *
from utils import *
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
        config_map = yaml.load(conf, Loader=yaml.FullLoader)
    return config_map, kwargs

def setup_opts():
    ## Parse command line args.
    parser = argparse.ArgumentParser(description="""Script to download and parse input files, and (TODO) run the  pipeline using them.""")
    # general parameters
    group = parser.add_argument_group('Main Options')
    group.add_argument('--config', type=str, default="/home/grads/haghani/SynVerse/code/"
                       "config_files/experiment_1/emlp_dsmiles_c1hot.yaml",
                       help="Configuration file for this script.")

    group.add_argument('--feat', type=str,
                       help="Put the name of the features to use, separated by space.")
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
    print('SYNVERSE STARTING')
    drug_features = params.drug_features
    cell_line_features = params.cell_line_features
    model_info = params.models
    splits = params.splits
    split_dir = params.split_dir
    synergy_file = inputs.processed_syn_file


    '''Read synergy triplets'''
    synergy_df = pd.read_csv(synergy_file, sep='\t', dtype={'drug_1_pid': str, 'drug_2_pid': str, 'cell_line_name': str})
    drug_pids = sorted(list(set(synergy_df['drug_1_pid']).union(set(synergy_df['drug_2_pid']))))
    cell_line_names = sorted(synergy_df['cell_line_name'].unique())


   #********************************** GET FEATURES READY *******************************************************
    ''' Read parsed drug features and do user-chosen preprocessing.'''
    dfeat_dict, dfeat_names = prepare_drug_features(drug_features, drug_pids, params, inputs)

    ''' Read parsed cell line features and do user-chosen preprocessing.'''
    cfeat_dict, cfeat_names = prepare_cell_line_features(cell_line_features, cell_line_names, params, inputs)

    '''Filter out the triplets based on the availability of drug and cell line features'''
    synergy_df = feature_based_filtering(synergy_df, dfeat_dict['mtx'], cfeat_dict['mtx'], params.feature)

    '''keep the cell lines consisting of at least 5% of the total #triplets in the final dataset.'''
    synergy_df = abundance_based_filtering(synergy_df, min_frac=0.05)
    print_synergy_stat(synergy_df)

    '''Rename column names to more generalized ones. Also, convert drug and cell line ids to numerical ids compatible with models.'''
    synergy_df, drug_2_idx, cell_line_2_idx = generalize_data(synergy_df,
                    col_name_map= {'drug_1_pid': 'source', 'drug_2_pid': 'target', 'cell_line_name': 'edge_type'})

    #convert 'pid' and 'cell_line_name' to numerical index in the feature dictionaries.
    for feat_name in dfeat_names:
        if isinstance(dfeat_dict['mtx'][feat_name], pd.DataFrame):
            cur_dfeat = dfeat_dict['mtx'][feat_name]
            cur_dfeat['idx'] = cur_dfeat['pid'].astype(str).apply(lambda x: drug_2_idx.get(x))
            cur_dfeat.drop_duplicates(subset=['pid'], inplace=True)
            cur_dfeat.dropna(subset=['idx'], inplace=True)
            cur_dfeat.set_index('idx', inplace=True)
            cur_dfeat.drop(axis=1, columns=['pid'], inplace=True)
            #sort drugs according to index
            cur_dfeat.sort_index(inplace=True)
            assert list(cur_dfeat.index) == list(range(len(cur_dfeat))), print('index not in order.')
            # save feature of drugs as numpy array
            dfeat_dict['mtx'][feat_name] = cur_dfeat.values

        elif isinstance(dfeat_dict['mtx'][feat_name], dict):
            dfeat_dict['mtx'][feat_name] = {drug_2_idx[str(old_key)]: value for old_key, value in
                                dfeat_dict['mtx'][feat_name].items() if old_key in drug_2_idx}

    for feat_name in cfeat_names:
        cur_cfeat = cfeat_dict['mtx'][feat_name]
        cur_cfeat['idx'] = cur_cfeat['cell_line_name'].astype(str).apply(lambda x: cell_line_2_idx.get(x))
        cur_cfeat.dropna(subset=['idx'], inplace=True)
        cur_cfeat.set_index('idx', inplace=True)
        cur_cfeat.drop(axis=1, columns=['cell_line_name'], inplace=True)
        cur_cfeat.sort_index(inplace=True)
        assert list(cur_cfeat.index) == list(range(len(cur_cfeat))), print(
            'index not in order.')
        cfeat_dict['mtx'][feat_name] = cur_cfeat.values

    #******************************************* MODEL TRAINING ***********************************************

    #***********************************************Figure out the feature combinations to train the model on ***
    #if I put --feat as user defined parameter, then I want to use only that combination of feature. override whatever is
    #given in the config
    use_feat = kwargs.get('feat')
    drug_cell_feat_combs = get_feature_comb_wrapper(dfeat_names, dfeat_dict, cfeat_names, cfeat_dict,
                             use_feat=use_feat, max_feat=params.max_feat)

    for (select_drug_feat, select_cell_feat) in drug_cell_feat_combs:
        #only keep the selected drug and cell feature for training and further analysis
        select_dfeat_dict = keep_selected_feat(dfeat_dict, select_drug_feat)
        select_cfeat_dict = keep_selected_feat(cfeat_dict, select_cell_feat)
        #depending on the selected encoders modify the model architecture here.
        select_model_info = get_select_model_info(model_info, select_dfeat_dict['encoder'], select_cfeat_dict['encoder'])

        ''' prepare split'''
        for split in splits:
            split_type = split['type']
            n_folds = split['n_folds']
            test_frac = split['test_frac']

            #split into train test
            split_prefix = split_dir + f'/{get_feat_prefix(params, dfeat_dict, cfeat_dict)}/k_{params.abundance}/'
            # split_prefix = split_dir + f'/k_{params.k}/'

            train_df, test_df = wrapper_train_test(synergy_df, split_type, test_frac, split_prefix, force_run=False)
            # del(synergy_df)
            #split into train_val for n_folds
            train_idx, val_idx = wrapper_nfold_split(train_df, split_type, n_folds, split_prefix, force_run=False)

            print('ran till model part')
            print('SPLIT: ', split_type)

            hyperparam = combine_hyperparams(select_model_info)
            best_n_epochs = params.epochs

            #todo: give explanatory outfile/directory names.
            out_file_prefix = create_file_prefix(params, select_dfeat_dict, select_cfeat_dict, split_type)

            # out_file_prefix = params.out_dir+'/test.txt'
            kwargs['split_type'] = split_type
            runner = Encode_MLP_runner(train_df, train_idx, val_idx, select_dfeat_dict, select_cfeat_dict,
                     out_file_prefix, params, select_model_info, device, **kwargs)

            if params.mode == 'hp_tune':
                # find best hyperparam setup
                hyperparam, best_n_epochs = runner.find_best_hyperparam(params.bohb['server_type'], **kwargs)
                # train the model with best hyperparam and both train and validation dataset
                trained_model_state, train_loss = runner.train_model_given_config(hyperparam, best_n_epochs)

            elif params.mode== 'train_val':
                trained_model_state, train_loss = runner.train_model_given_config(hyperparam, best_n_epochs,
                                                                                  validation=True)
            elif params.mode == 'train':
                # train the model with best hyperparam and both train and validation dataset
                trained_model_state, train_loss = runner.train_model_given_config(hyperparam, best_n_epochs)

            # evaluate model on test data
            test_loss = runner.get_test_score(test_df, trained_model_state, hyperparam, best_n_epochs)

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
        params.wandb = config_map['input_settings']['wandb']
        params.bohb = config_map['input_settings']['bohb']
        params.drug_chemprop_dir = input_dir + '/drug/chemprop/'
        params.out_dir = output_dir
        params.split_dir = input_dir + 'splits'
        run_SynVerse(inputs, params, **kwargs)


if __name__ == "__main__":
    config_map, kwargs = parse_args()
    main(config_map, **kwargs)




