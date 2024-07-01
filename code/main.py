import pandas as pd
from evaluation.split_generalized import *
from utils import *
import types
import argparse
from models.mlp_runner import *
from models.encoder_mlp_runner import *
from preprocess import *

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
                       "config_files/experiment_1/emlp_dgraph_c1hot.yaml",
                       help="Configuration file for this script.")
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
    models_info = params.models
    splits = params.splits
    split_dir = params.split_dir
    synergy_file = inputs.processed_syn_file

    # make sure that whatever features you preprocess they are checked out, as in, if any feature
    # needs encoder and the encoder is not mentioned then raise an error message.
    # check_config_compeleteness('drug', drug_features, models_info)
    # check_config_compeleteness('cell', cell_line_features, models_info)

    '''Read synergy triplets'''
    synergy_df = pd.read_csv(synergy_file, sep='\t', dtype={'drug_1_pid': str, 'drug_2_pid': str, 'cell_line_name': str})
    drug_pids = sorted(list(set(synergy_df['drug_1_pid']).union(set(synergy_df['drug_2_pid']))))
    cell_line_names = sorted(synergy_df['cell_line_name'].unique())


   #********************************** GET FEATURES READY *******************************************************
    ''' Read parsed drug features and do user-chosen preprocessing.'''
    dfeat_dict = {}
    dfeat_dim_dict={}
    dfeat_names = [f['name'] for f in drug_features]

    if ('1hot' in dfeat_names):
        one_hot_feat= pd.DataFrame(np.eye(len(drug_pids)))
        one_hot_feat['pid'] = drug_pids
        dfeat_dict['1hot'] = one_hot_feat
        dfeat_dim_dict['1hot'] = one_hot_feat.shape[1]-1

    if 'MACCS' in dfeat_names:
        chem_prop_dir = params.drug_chemprop_dir
        maccs_file = chem_prop_dir + 'MACCS.tsv'
        maccs_df = pd.read_csv(maccs_file,dtype={'pid':str}, sep='\t', index_col=None)

        #TODO: if any preprocessing step is mentioned for 'MACCS' feature, do that here.
        dfeat_dict['MACCS'] = maccs_df
        dfeat_dim_dict['MACCS'] = maccs_df.shape[1]-1

    if 'smiles' in dfeat_names:
        smiles_file = inputs.drug_smiles_file
        smiles_df = pd.read_csv(smiles_file,dtype={'pid':str}, sep='\t', index_col=None)
        max_len = params.models[0]['hp']['max_seq_length']
        smiles_df, vocab_size = get_vocab_smiles(smiles_df, max_len)
        dfeat_dict['smiles'] = smiles_df
        dfeat_dim_dict['smiles'] = vocab_size

    if 'MFP' in dfeat_names:
        chem_prop_dir = params.drug_chemprop_dir
        mfp_file = chem_prop_dir + 'Morgan_fingerprint.tsv'
        mfp_df = pd.read_csv(mfp_file,dtype={'pid':str}, sep='\t', index_col=None)
        #TODO: if any preprocessing step is mentioned for 'MACCS' feature, do that here.
        dfeat_dict['MFP'] = mfp_df
        dfeat_dim_dict['MFP'] = mfp_df.shape[1]-1

    if 'ECFP_4' in dfeat_names:
        chem_prop_dir = params.drug_chemprop_dir
        ecfp_file = chem_prop_dir + 'ECFP_4.tsv'
        ecfp_df = pd.read_csv(ecfp_file,dtype={'pid':str}, sep='\t', index_col=None)
        #TODO: if any preprocessing step is mentioned for 'MACCS' feature, do that here.
        dfeat_dict['ECFP_4'] = ecfp_df
        dfeat_dim_dict['ECFP_4'] = ecfp_df.shape[1]-1


    if 'mol_graph' in dfeat_names:
        mol_graph_file = inputs.drug_graph_file
        with open(mol_graph_file, 'rb') as file:
            pid_to_adjacency_mol_feat = pickle.load(file)
        mol_pyg_dict, mol_feat_dim = mol_graph_to_GCN_data(pid_to_adjacency_mol_feat)
        dfeat_dict['mol_graph'] = mol_pyg_dict
        dfeat_dim_dict['mol_graph'] = mol_feat_dim

        #convert mol_graph to datatype compatible with GCN/GAT

    ''' Read parsed cell line features and do user-chosen preprocessing.'''
    cfeat_dict = {}
    cfeat_dim_dict={}
    cfeat_names = [f['name'] for f in cell_line_features]
    cfeat_preprocess= {f['name']:f['preprocess'] for f in cell_line_features}
    cfeat_norm= {f['name']:f['norm'] for f in cell_line_features}

    if ('1hot' in cfeat_names):
        one_hot_feat = pd.DataFrame(np.eye(len(cell_line_names)))
        one_hot_feat['cell_line_name'] = cell_line_names
        cfeat_dict['1hot'] = one_hot_feat
        cfeat_dim_dict['1hot'] = one_hot_feat.shape[1]-1

    if 'genex' in cfeat_names:
        feat_name = 'genex'
        ccle_file = inputs.cell_line_file
        ccle_df = pd.read_csv(ccle_file, sep='\t')

        #do any preprocessing
        if cfeat_preprocess['genex']=='lincs_1000':
            feat_name = feat_name+'_lincs_1000'
            ccle_df = landmark_gene_filter(ccle_df, inputs.lincs)
        #do any normalization
        if cfeat_norm['genex'] is not None:
            feat_name =feat_name+'_' + cfeat_norm['genex']
            ccle_df.set_index('cell_line_name', inplace=True)
            ccle_df, means, std = normalize(ccle_df, norm_type=cfeat_norm['genex'])
            ccle_df.reset_index(names='cell_line_name', inplace=True)

        cfeat_dict[feat_name] = ccle_df
        cfeat_dim_dict[feat_name] = ccle_df.shape[1]-1

    '''Filter out the triplets based on the availability of drug and cell line features'''
    synergy_df = feature_based_filtering(synergy_df, dfeat_dict, cfeat_dict, params.feature)

    '''keep the cell lines consisting of at least 1% of the total #triplets in the final dataset.'''
    synergy_df = abundance_based_filtering(synergy_df, min_frac=0.05)
    print_synergy_stat(synergy_df)

    '''Rename column names to more generalized ones. Also, convert drug and cell line ids to numerical ids compatible with models.'''
    synergy_df, drug_2_idx, cell_line_2_idx = generalize_data(synergy_df,
                    col_name_map= {'drug_1_pid': 'source', 'drug_2_pid': 'target', 'cell_line_name': 'edge_type'})

    #convert 'pid' and 'cell_line_name' to numerical index in the feature dictionaries.
    for feat_name in dfeat_dict:
        if isinstance(dfeat_dict[feat_name], pd.DataFrame):
            dfeat_dict[feat_name]['idx'] = dfeat_dict[feat_name]['pid'].astype(str).apply(lambda x: drug_2_idx.get(x))
            dfeat_dict[feat_name].drop_duplicates(subset=['pid'], inplace=True)
            dfeat_dict[feat_name].dropna(subset=['idx'], inplace=True)
            dfeat_dict[feat_name].set_index('idx', inplace=True)
            dfeat_dict[feat_name].drop(axis=1, columns=['pid'], inplace=True)
            #sort drugs according to index
            dfeat_dict[feat_name].sort_index(inplace=True)
            assert list(dfeat_dict[feat_name].index) == list(range(len(dfeat_dict[feat_name]))), print('index not in order.')
            # save feature of drugs as numpy array
            dfeat_dict[feat_name] = dfeat_dict[feat_name].values

        elif isinstance(dfeat_dict[feat_name], dict):
            dfeat_dict[feat_name] = {drug_2_idx[str(old_key)]: value for old_key, value in
                                dfeat_dict[feat_name].items() if old_key in drug_2_idx}


    for feat_name in cfeat_dict:
        cfeat_dict[feat_name]['idx'] = cfeat_dict[feat_name]['cell_line_name'].astype(str).apply(lambda x: cell_line_2_idx.get(x))
        cfeat_dict[feat_name].dropna(subset=['idx'], inplace=True)
        cfeat_dict[feat_name].set_index('idx', inplace=True)
        cfeat_dict[feat_name].drop(axis=1, columns=['cell_line_name'], inplace=True)
        cfeat_dict[feat_name].sort_index(inplace=True)
        assert list(cfeat_dict[feat_name].index) == list(range(len(cfeat_dict[feat_name]))), print(
            'index not in order.')
        cfeat_dict[feat_name] = cfeat_dict[feat_name].values

    #******************************************* MODEL TRAINING ***********************************************

    #***********************************************Figure out the feature combinations to train the model on ***
    drug_feat_combs = compute_feature_combination(drug_features, params.max_feat)
    cell_feat_combs = compute_feature_combination(cell_line_features, params.max_feat)
    drug_cell_feat_combs = find_drug_cell_feat_combs(drug_feat_combs, cell_feat_combs)

    for (select_drug_feat, select_cell_feat) in drug_cell_feat_combs:
        select_dfeat_dict = {feat: dfeat_dict[feat] for feat in select_drug_feat }
        select_dfeat_dim_dict = {feat: dfeat_dim_dict[feat] for feat in select_drug_feat }
        select_cfeat_dict = {feat: cfeat_dict[feat] for feat in select_cell_feat }
        select_cfeat_dim_dict = {feat: cfeat_dim_dict[feat] for feat in select_cell_feat }

        ''' prepare split'''
        for split in splits:
            split_type=split['type']
            n_folds = split['n_folds']
            test_frac= split['test_frac']

            #split into train test
            split_prefix = split_dir + f'/{get_feat_prefix(params, dfeat_dict, cfeat_dict)}/k_{params.abundance}/'
            # split_prefix = split_dir + f'/k_{params.k}/'

            train_df, test_df = wrapper_train_test(synergy_df, split_type, test_frac, split_prefix, force_run=False)
            # del(synergy_df)
            #split into train_val for n_folds
            train_idx, val_idx = wrapper_nfold_split(train_df, split_type, n_folds, split_prefix, force_run=False)

            print('ran till model part')
            print('SPLIT: ', split_type)
            for model_info in models_info:
                model_name = model_info['name']
                hyperparam = model_info['hp'] #default hyperpam unless best hyperparm is computed by hyperparam tuning
                best_n_epochs = model_info['epochs']

                out_file_prefix = create_file_prefix(params, select_drug_feat, select_cell_feat, model_name, split_type)
                if model_name=='MLP':
                    print(out_file_prefix)
                    #initialize the runner class
                    runner = MLP_runner(split_type, train_df, train_idx, val_idx, select_dfeat_dict, select_cfeat_dict,
                                    select_dfeat_dim_dict, select_cfeat_dim_dict,
                                    out_file_prefix, params, model_info, device, **kwargs) #each runner initiate an MLP model.

                elif model_name=='Encoder_MLP':
                    # kwargs['drug_encoder'] = model_info['drug_encoder']
                    # kwargs['cell_encoder'] = model_info['cell_encoder']
                    runner = Encode_MLP_runner(split_type, train_df, train_idx, val_idx, select_dfeat_dict, select_cfeat_dict,
                                               select_dfeat_dim_dict, select_cfeat_dim_dict, out_file_prefix, params, model_info, device, **kwargs)

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
        params.splits = config_map['input_settings']['splits']
        params.feature = config_map['input_settings']['feature']
        params.abundance = config_map['input_settings']['abundance']
        params.max_feat=config_map['input_settings']['max_feat']
        params.mode=config_map['input_settings']['mode']
        input_settings = config_map.get('input_settings', {})
        params.wandb= types.SimpleNamespace(**input_settings.get('wandb', {}))
        params.bohb = config_map['input_settings']['bohb']
        params.drug_chemprop_dir = input_dir + '/drug/chemprop/'
        params.out_dir = output_dir
        params.split_dir = input_dir + 'splits'
        run_SynVerse(inputs, params, **kwargs)


if __name__ == "__main__":
    config_map, kwargs = parse_args()
    main(config_map, **kwargs)




