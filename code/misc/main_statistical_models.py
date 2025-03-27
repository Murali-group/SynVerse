from network_rewire import *
import types
import argparse
from preprocessing.cell_line_preprocess import *
from preprocessing.drug_preprocess import *
from analysis.statistical_synergy_prediction_model import *
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
                       "config_files/experiment_1/debug_statistical.yaml",
                       help="Configuration file for this script.")
    # group.add_argument('--score_name', type=str, default='synergy_loewe_mean', help="Name of the score to predict.")
    group.add_argument('--score_name', type=str, default='S_mean_mean', help="Name of the score to predict.")

    group.add_argument('--feat', type=str,
                       help="Put the name of the features to use, separated by space. Applicable when you want to run just one set of features.")
    group.add_argument('--split', type=str,
                       help="Put the name of the split types to run, separated by space.")
    group.add_argument('--start_run', type=int, help='From which run should the model start from. This is to help when'
                    'some model has been trained for first 2 runs but the terminated by arc. Then next time we need to start from run 2, hence start_run should be 2', default=0)
    group.add_argument('--end_run', type=int, help='How many runs you want. end_run=5 means we will get runs starting at start_run and ending at (end_run-1)', default=5)


    return parser


def run_SynVerse(inputs, params, **kwargs):
    #TODO: set default values for the params if not given in config file.
    print(device)
    print('SYNVERSE STARTING')
    drug_features = params.drug_features
    cell_line_features = params.cell_line_features
    abundance=params.abundance
    model_info = params.models
    splits = params.splits
    split_dir = params.split_dir
    out_dir = params.out_dir

    # score_name = 'synergy_loewe_mean' #synergy score to use
    # score_name = 'S_mean_mean' #synergy score to use
    score_name=kwargs.get('score_name')
    if score_name == 'S_mean_mean':
        synergy_file = inputs.smean_processed_syn_file
    elif score_name == 'synergy_loewe_mean':
        synergy_file = inputs.loewe_processed_syn_file

    '''Read synergy triplets'''
    synergy_df = pd.read_csv(synergy_file, sep='\t', dtype={'drug_1_pid': str, 'drug_2_pid': str, 'cell_line_name': str})
    drug_pids = sorted(list(set(synergy_df['drug_1_pid']).union(set(synergy_df['drug_2_pid']))))
    cell_line_names = sorted(synergy_df['cell_line_name'].unique())

    # synergy_df.to_csv(f'test_{score_name}.tsv', sep='\t', index=False)

   #********************************** GET FEATURES READY *******************************************************
    ''' Read parsed drug features and do user-chosen filtering and preprocessing.'''
    dfeat_dict, dfeat_names = prepare_drug_features(drug_features, drug_pids, params, inputs, device)

    ''' Read parsed cell line features and do user-chosen filtering and preprocessing.'''
    cfeat_dict, cfeat_names = prepare_cell_line_features(cell_line_features, cell_line_names, inputs)


    '''Filter out the triplets based on the availability of drug and cell line features'''
    synergy_df = feature_based_filtering(synergy_df, dfeat_dict['value'], cfeat_dict['value'], params.feature)

    '''keep the cell lines consisting of at least abundance% of the total #triplets in the final dataset.'''
    synergy_df = abundance_based_filtering(synergy_df, min_frac=abundance)

    # plot_dist(synergy_df[score_name], out_dir= f'{params.input_dir}/stat/{get_feat_prefix(dfeat_dict, cfeat_dict)}_k_{abundance}_{score_name}')



    #******************************************* MODEL TRAINING ***********************************************

    #***********************************************Figure out the feature combinations to train the model on ***
    #if I put --feat as user defined parameter, then I want to use only that combination of feature. override whatever is
    #given in the config
    use_feat = kwargs.get('feat')


    start_run = kwargs.get('start_run')
    end_run = kwargs.get('end_run')

    ''' prepare split'''
    mse_loss_dict = {}
    rmse_loss_dict = {}
    pearsons_dict = {}
    pval_dict = {}

    for split in splits:
        split_type = split['type']
        # n_folds = split['n_folds']
        test_frac = split['test_frac']
        val_frac = split['val_frac']

        for run_no in range(start_run, end_run):
            #if user defined split type is present as kwargs param, then only the split types common between config and kwargs param
            #will run.
            udef_split_types = kwargs.get('split')
            if udef_split_types is not None:
                udef_split_types = udef_split_types.split(' ')
                if split_type not in udef_split_types: #do not run split type not present in kwargs param
                    continue

            split_feat_str = get_feat_prefix(dfeat_dict, cfeat_dict)
            split_info_str = f"/{split_feat_str}/k_{abundance}_{score_name}/{split_type}_{test_frac}_{val_frac}/run_{run_no}/"


            split_file_path = split_dir + split_info_str


            force_split = False

            #split into train test val
            test_df, all_train_df, train_idx, val_idx, drug_2_idx, cell_line_2_idx = wrapper_test_train_val(copy.deepcopy(synergy_df), split_type, test_frac, val_frac, split_file_path, seed=run_no, force_run=force_split)
            # print('\n\nedge type sepcific node_degree_based_model')
            # predicted, mse_loss, pearson_cor, pvalue = edge_type_spec_node_degree_based_avg_model(all_train_df, test_df, score_name)

            # print('\n\nedge type sepcific node_degree_based_model')
            # predicted, mse_loss, pearson_cor, pvalue = edge_type_spec_node_degree_based_sampling_model(all_train_df, test_df, score_name)

            # print('\n\nnode_degree_based_sampling model')
            # predicted, mse_loss, pearson_cor, pvalue = node_degree_based_sampling_model(all_train_df, test_df, score_name)

            # print('\n\nnode_degree_based_sampling model')
            # predicted, mse_loss, pearson_cor, pvalue = edge_type_spec_node_degree_based_model(all_train_df, test_df, score_name, choice='average')

            # print("\n\nlabel_distribution_based_model")
            predicted, mse_loss, pearson_cor, pvalue = edge_type_label_distribution_based_model(all_train_df, test_df, score_name, split_type=split_type)
            mse_loss_dict[run_no] = mse_loss
            rmse_loss_dict[run_no] = np.sqrt(mse_loss)
            pearsons_dict[run_no] =pearson_cor
            pval_dict[run_no] = pvalue

            # print('Split: ', split_type, 'Run:', run_no)
            # print('RMSE: ', rmse_loss_dict[run_no])
            # print('PEARSONS: ', pearson_cor)
            #


        avg_mse = np.mean(list(mse_loss_dict.values()))
        avg_rmse = np.mean(list(rmse_loss_dict.values()))
        avg_pearsons = np.mean(list(pearsons_dict.values()))
        avg_pval = np.mean(list(pval_dict.values()))



        print(split_type)
        print(f'Average MSE {avg_mse:.4f}')
        print(f'Average RMSE {avg_rmse:.4f}')
        print(f'Average PEARSONS {avg_pearsons:.4f}')
        print(f'Average PEARSONS {avg_pval:.4f}')







def main(config_map, **kwargs):


    # config_map = load_yaml_file(config_map)
    input_dir = config_map['input_settings']['input_dir']
    output_dir = config_map['output_settings']['output_dir']

    inputs = types.SimpleNamespace()
    params = types.SimpleNamespace()

    inputs.smean_processed_syn_file = input_dir + 'synergy/synergy_scores_S_mean_mean.tsv' #manually renamed previous synergy_scores.tsv (on which I had all the runs till Decemeber 11, 2024) to synergy_scores_S_mean_mean.tsv It is the same as new S_mean_synergy_zip_std_threshold_0.1.tsv.
    inputs.loewe_processed_syn_file = input_dir + 'synergy/synergy_loewe_S_mean_std_threshold_0.1.tsv'
    # inputs.processed_syn_file = input_dir + 'synergy/merged.tsv'

    inputs.drug_smiles_file = input_dir + 'drug/smiles.tsv'
    inputs.drug_graph_file = input_dir + 'drug/molecular_graph.pickle'
    inputs.drug_target_file = input_dir + 'drug/target.tsv'


    inputs.cell_line_file = input_dir + 'cell-line/gene_expression.tsv'
    inputs.lincs = input_dir + 'cell-line/LINCS_1000.txt'

    params.drug_features = config_map['input_settings']['drug_features']
    params.cell_line_features = config_map['input_settings']['cell_line_features']
    params.models = config_map['input_settings']['models']
    params.epochs = config_map['input_settings']['epochs']
    params.autoencoder_dims = [[1024, 512], [512, 256], [256, 128], [256, 64]]

    params.splits = config_map['input_settings']['splits']
    params.feature = config_map['input_settings']['feature']
    params.abundance = config_map['input_settings']['abundance']
    params.max_feat=config_map['input_settings']['max_feat']
    params.hp_tune=config_map['input_settings']['hp_tune']
    params.train_mode = config_map['input_settings']['train_mode']
    params.rewire = config_map['input_settings'].get('rewire', False)
    params.rewire_method = config_map['input_settings'].get('rewire_method', None)
    params.shuffle = config_map['input_settings'].get('shuffle', False) #shuffle/randomize features
    params.sample_norm = config_map['input_settings'].get('sample_norm', False) #sample triplets to maintain a normal distribution
    params.retain_ratio = config_map['input_settings'].get('retain_ratio', 0.99) #sample triplets to maintain a normal distribution

    params.batch_size = config_map['input_settings'].get('batch_size', 4096)
    input_settings = config_map.get('input_settings', {})
    params.wandb = types.SimpleNamespace(**input_settings.get('wandb', {}))
    params.bohb = config_map['input_settings']['bohb']
    params.drug_chemprop_dir = input_dir + '/drug/chemprop/'
    params.input_dir= input_dir
    params.out_dir = output_dir
    params.split_dir = input_dir + 'splits'
    run_SynVerse(inputs, params, **kwargs)


if __name__ == "__main__":
    config_map, kwargs = parse_args()
    main(config_map, **kwargs)




