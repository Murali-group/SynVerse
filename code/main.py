import copy
import os.path
import pandas as pd
from evaluation.split import *
from graph_utils import *
from utils import *
from plots.plot_utils import *
import types
import argparse
from models.encoder_mlp_runner import *
from train_ae import *
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
                       "config_files/experiment_1/finetuned_model.yaml",
                       help="Configuration file for this script.")
    group.add_argument('--score_name', type=str, default='S_mean_mean', help="Name of the score to predict.")
    group.add_argument('--feat', type=str,
                       help="Put the name of the features to use, separated by space. Applicable when you want to run just one set of features.")
    group.add_argument('--split', type=str,
                       help="Put the name of the split types to run, separated by space.")
    group.add_argument('--start_run', type=int, help='From which run should the model start from. This is to help when'
                    'some model has been trained for first 2 runs but the terminated by arc. Then next time we need to start from run 2, hence start_run should be 2', default=0)
    group.add_argument('--end_run', type=int, help='How many runs you want. end_run=5 means we will get runs starting at start_run and ending at (end_run-1)', default=5)

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

    plot_dist(synergy_df[score_name], out_dir= f'{params.input_dir}/stat/{get_feat_prefix(dfeat_dict, cfeat_dict)}_k_{abundance}_{score_name}')


    if params.sample_norm: #sample the triplets from synergy_df such that they follow a normal distribution.
        retain_ratio = params.retain_ratio #how much sample to retain
        synergy_df = filter_normal_conforming_data(synergy_df, score_name,retain_ratio=retain_ratio)
        split_dir = os.path.join(split_dir, f'sample_norm_{retain_ratio}')
        out_dir = os.path.join(out_dir, f'sample_norm_{retain_ratio}')

    #******************************************* MODEL TRAINING ***********************************************

    #***********************************************Figure out the feature combinations to train the model on ***
    #if I put --feat as user defined parameter, then I want to use only that combination of feature. override whatever is
    #given in the config
    use_feat = kwargs.get('feat')
    drug_cell_feat_combs = get_feature_comb_wrapper(dfeat_names, dfeat_dict, cfeat_names, cfeat_dict,
                            use_feat=use_feat, max_drug_feat=params.max_drug_feat,
                            min_drug_feat = params.min_drug_feat, max_cell_feat=params.max_cell_feat, min_cell_feat = params.min_cell_feat)


    start_run = kwargs.get('start_run')
    end_run = kwargs.get('end_run')

    data_distribution_dict = {'run':[], 'split': [], 'train_mean':[], 'train_std':[], 'test_mean':[], 'test_std':[]}
    ''' prepare split'''
    for run_no in range(start_run, end_run):
        for split in splits:
            split_type = split['type']
            # n_folds = split['n_folds']
            test_frac = split['test_frac']
            val_frac = split['val_frac']
            print('SPLIT: ', split_type)

            #if user defined split type is present as kwargs param, then only the split types common between config and kwargs param
            #will run.
            udef_split_types = kwargs.get('split')
            if udef_split_types is not None:
                udef_split_types = udef_split_types.split(' ')
                if split_type not in udef_split_types: #do not run split type not present in kwargs param
                    continue

            split_feat_str = get_feat_prefix(dfeat_dict, cfeat_dict)
            split_info_str = f"/{split_feat_str}/k_{abundance}_{score_name}/{split_type}_{test_frac}_{val_frac}/run_{run_no}/"


            print('SPLIT STR: ', split_info_str)
            split_file_path = split_dir + split_info_str


            force_split = False

            #split into train test val
            test_df, all_train_df, train_idx, val_idx, drug_2_idx, cell_line_2_idx = wrapper_test_train_val(copy.deepcopy(synergy_df), split_type, test_frac, val_frac, split_file_path, seed=run_no, force_run=force_split)

            # #plot synergy score distribution for train and test set
            if score_name=='synergy_loewe_mean':
                plot_double_dist(all_train_df['synergy_loewe_mean'], all_train_df['S_mean_mean'], labels = ['Loewe', 'S_mean'], prefix=f'train', out_dir=split_file_path)
                plot_double_dist(test_df['synergy_loewe_mean'], test_df['S_mean_mean'], labels = ['Loewe', 'S_mean'], prefix= f'test', out_dir=split_file_path)

            plot_dist(all_train_df[score_name], f'train_{score_name}', out_dir=split_file_path)
            plot_dist(test_df[score_name], f'test_{score_name}', out_dir=split_file_path)
            data_distribution_dict['run'].append(run_no)
            data_distribution_dict['split'].append(split)
            data_distribution_dict['train_mean'].append(np.mean(all_train_df[score_name]))
            data_distribution_dict['train_std'].append(np.std(all_train_df[score_name]))
            data_distribution_dict['test_mean'].append(np.mean(test_df[score_name]))
            data_distribution_dict['test_std'].append(np.std(test_df[score_name]))

            #convert feature dataframes into numpy arrays while in the array row i corresponds to the drug with numerical idx i
            cur_dfeat_dict = copy.deepcopy(dfeat_dict)
            cur_cfeat_dict = copy.deepcopy(cfeat_dict)
            cur_dfeat_dict['value'], cur_cfeat_dict['value'] = get_index_sorted_feature_matrix(cur_dfeat_dict['value'], drug_2_idx,
                                                   cur_cfeat_dict['value'], cell_line_2_idx)


            # Reduce dimension of data or compress data using autoenencoder
            train_drug_idx = list(set(all_train_df['source']).union(set(all_train_df['target'])))
            train_cell_idx = list(set(all_train_df['edge_type']).union(set(all_train_df['edge_type'])))

            cur_dfeat_dict['value'], cur_dfeat_dict['dim'] = autoencoder_wrapper(cur_dfeat_dict['value'],cur_dfeat_dict['dim'], cur_dfeat_dict['compress'],
                                                          train_drug_idx, hidden_dim_options=params.autoencoder_dims, epoch=500,
                                                          file_prefix=f'{params.input_dir}/drug/AE/{split_info_str}/', device=device, force_run=force_split)
            cur_cfeat_dict['value'], cur_cfeat_dict['dim'] = autoencoder_wrapper(cur_cfeat_dict['value'], cur_cfeat_dict['dim'], cur_cfeat_dict['compress'],
                                                          train_cell_idx, hidden_dim_options=params.autoencoder_dims, epoch=500,
                                                          file_prefix=f'{params.input_dir}/cell-line/AE/{split_info_str}/',
                                                          device=device, force_run=force_split)

            # Normalize data based on training data. Use the computed mean, std from training data to normalize test data.
            cur_dfeat_dict['value'], cur_cfeat_dict['value'] = normalization_wrapper(cur_dfeat_dict['value'],
                                                                                     cur_cfeat_dict['value'],
                                                                                     cur_dfeat_dict['norm'],
                                                                                     cur_cfeat_dict['norm'],
                                                                                     all_train_df)

            all_train_df = all_train_df[['source', 'target','edge_type', score_name]]

            for (select_drug_feat, select_cell_feat) in drug_cell_feat_combs:
                print('drug and cell line features in use: ', select_drug_feat, select_cell_feat)

                # only keep the selected drug and cell feature for training and further analysis
                select_dfeat_dict = keep_selected_feat(cur_dfeat_dict, select_drug_feat)
                select_cfeat_dict = keep_selected_feat(cur_cfeat_dict, select_cell_feat)
                # depending on the selected encoders modify the model architecture here.
                select_model_info = get_select_model_info(model_info, select_dfeat_dict['encoder'], select_cfeat_dict['encoder'])

                hyperparam = combine_hyperparams(select_model_info)
                given_epochs = params.epochs
                kwargs['split_type'] = split_type
                out_file_prefix = create_file_prefix(out_dir,abundance, select_dfeat_dict, select_cfeat_dict, split_type,
                                                     score_name, split_feat_str=split_feat_str, run_no=run_no)

                if params.rewire:
                    # rewire the training dataset keeping the node degree intact. Modified all_train_df, train_idx, val_idx
                    rewire_methods = params.rewire_method  # SA => Simulated annealing, SM=> Sneppen-Maslov, RS: Rubinov as randomization method

                    for rewire_method in rewire_methods:
                        for rand_net in range(10): #run model on 10 randmized network
                            print(f'Running model on {rand_net}th randomized network')
                            rewired_all_train_df, rewired_train_idx, rewired_val_idx = get_rewired_train_val(all_train_df, score_name, rewire_method,
                                                                split_type, val_frac, out_dir=f'{split_file_path}{rand_net}',
                                                                force_run=force_split)
                            wrapper_plot_difference_in_degree_distribution(rewired_all_train_df, all_train_df, score_name, cell_line_2_idx, plot_file_prefix = f'{split_file_path}/{rand_net}_{rewire_method}'  )

                            out_file_prefix_rand = f'{out_file_prefix}_rewired_{rand_net}_{rewire_method}'
                            runner = Encode_MLP_runner(rewired_all_train_df, rewired_train_idx, rewired_val_idx, select_dfeat_dict,
                                                       select_cfeat_dict,
                                                       out_file_prefix_rand, params, select_model_info, device, **kwargs)

                            if params.train_mode['use_best_hyperparam']:
                                # find the best hyperparam saved in a file for the given features and architecture
                                hyperparam, _ = extract_best_hyperparam(out_file_prefix + '_best_hyperparam.txt')  # incase rewire is true, we want to find the best param for the main model without rewiring.

                            trained_model_state, train_loss = runner.train_model_given_config(hyperparam, given_epochs,
                                                                                              validation=True,
                                                                                              save_output=True)  # when validation=True, use given epochs as you can always early stop using validation loss
                            runner.get_test_score(test_df, trained_model_state, hyperparam, save_output=True,
                                                  file_prefix='_val_true_')

                elif params.shuffle:
                    # shuffle the features in training dataset keeping the test intact
                    for shuffle_no in range(10): #run model on 10 randmized network

                        shuffled_dfeat_dict = copy.deepcopy(select_dfeat_dict)
                        shuffled_cfeat_dict = copy.deepcopy(select_cfeat_dict)


                        shuffled_dfeat_dict['value'] = shuffle_features(shuffled_dfeat_dict['value'])
                        shuffled_cfeat_dict['value'] = shuffle_features(shuffled_cfeat_dict['value'])

                        out_file_prefix_shuffle = f'{out_file_prefix}_shuffled_{shuffle_no}'
                        runner = Encode_MLP_runner(all_train_df, train_idx, val_idx, shuffled_dfeat_dict,
                                                   shuffled_cfeat_dict,
                                                   out_file_prefix_shuffle, params, select_model_info, device, **kwargs)
                        if params.train_mode['use_best_hyperparam']:
                            # find the best hyperparam saved in a file for the given features and architecture
                            hyperparam, _ = extract_best_hyperparam(out_file_prefix + '_best_hyperparam.txt')

                        trained_model_state, train_loss = runner.train_model_given_config(hyperparam, given_epochs,
                                                                                          validation=True,
                                                                                          save_output=True)  # when validation=True, use given epochs as you can always early stop using validation loss

                        runner.get_test_score(test_df, trained_model_state, hyperparam, save_output=True,
                                              file_prefix='_val_true_')



                else:
                    runner = Encode_MLP_runner(all_train_df, train_idx, val_idx, select_dfeat_dict, select_cfeat_dict,
                             out_file_prefix, params, select_model_info, device, **kwargs)

                    if params.hp_tune:
                        # find best hyperparam setup
                        runner.find_best_hyperparam(params.bohb['server_type'], **kwargs)

                    if params.train_mode['use_best_hyperparam']:
                        if params.sample_norm: #find the best param tuned  on 'not-sampled' data
                            hyperparam, _ = extract_best_hyperparam(out_file_prefix.replace(f'sample_norm_{retain_ratio}/','') + '_best_hyperparam.txt')
                        else:
                            #find the best hyperparam saved in a file for the given features and architecture
                            hyperparam_file = out_file_prefix+'_best_hyperparam.txt'

                            if abundance==0:  #if abundance==0, then use hyperparam tuned for k=0.05
                                hyperparam_file = hyperparam_file.replace('k_0', 'k_0.05')
                            hyperparam, _ = extract_best_hyperparam(hyperparam_file)

                    trained_model_state, train_loss = runner.train_model_given_config(hyperparam, given_epochs,validation=True,save_output=True) #when validation=True, use given epochs as you can always early stop using validation loss
                    runner.get_test_score(test_df, trained_model_state, hyperparam, save_output=True, file_prefix='_val_true_')


            del cur_dfeat_dict
            del cur_cfeat_dict
    data_distribution_df = pd.DataFrame(data_distribution_dict)
    data_distribution_df.to_csv(f'{params.input_dir}/stat/{get_feat_prefix(dfeat_dict, cfeat_dict)}_k_{abundance}_{score_name}.tsv', sep='\t', index=False)

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

        inputs.smean_processed_syn_file = input_dir + 'synergy/synergy_scores_S_mean_mean.tsv' #manually renamed previous synergy_scores.tsv (on which I had all the runs till Decemeber 11, 2024) to synergy_scores_S_mean_mean.tsv It is the same as new S_mean_synergy_zip_std_threshold_0.1.tsv.
        inputs.loewe_processed_syn_file = input_dir + 'synergy/synergy_loewe_S_mean_std_threshold_0.1.tsv'

        inputs.net_file = '/home/grads/tasnina/Projects/SynVerse/datasets/network/STRING/9606.protein.links.v12.0.txt.gz'
        inputs.prot_info_file = '/home/grads/tasnina/Projects/SynVerse/datasets/network/STRING/9606.protein.info.v12.0.txt.gz'

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
        params.abundance = config_map['input_settings'].get('abundance',0)

        params.max_drug_feat=config_map['input_settings'].get('max_drug_feat', 1)
        params.max_cell_feat=config_map['input_settings'].get('max_cell_feat', 1)

        params.min_drug_feat=config_map['input_settings'].get('min_drug_feat', 1)
        params.min_cell_feat=config_map['input_settings'].get('min_cell_feat', 1)

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




