"""
1. from config file get which models to run

2. call the function for cross validation folds on training labels 
    - number of folds can be different
    - Type of cross validation: 1. Leave drug combination out in all cell line 2. Random Cross Validation 
                          3. Leave one-drug out 4.Leave one-cell line out
    - Make sure to train all the models on same five-fold splits


3. call the function for minibatch creation depending on the ML model.

"""
import argparse
import yaml
#from tqdm import tqdm
#from scipy import sparse
import utils
from utils import *
import pickle
import matplotlib.pyplot as plt
import models.gnn.run_synverse_tissuenet as synverse_tissuenet
import pandas as pd
import numpy as np
import os
from data_split import cross_validation as cross_val


# import evaluation.evaluation_handler as evaluation_handler

def parse_args():
    parser = setup_opts()
    args = parser.parse_args()
    kwargs = vars(args)
    with open(args.config, 'r') as conf:
        #config_map = yaml.load(conf, Loader=yaml.FullLoader)
        config_map = yaml.load(conf)
    # TODO check to make sure the inputs are correct in config_map

    return config_map, kwargs


def setup_opts():
    ## Parse command line args.
    parser = argparse.ArgumentParser(description="""Script to download and parse input files,
                                     and (TODO) run the  pipeline using them.""")

    # general parameters
    group = parser.add_argument_group('Main Options')
    group.add_argument('--config', type=str, default="/home/tasnina/Projects/SynVerse/code/config-files/master_config_csbgpu3.yaml",
                       help="Configuration file for this script.")

    # group.add_argument('--config', type=str, default="/home/grads/tasnina/Projects/SynVerse/code/config-files/master_config_csbgpu5.yaml",
    #                    help="Configuration file for this script.")

    #GENE Expression param
    group.add_argument('--exp-score', type=str, default='Z_SCORE',
                       help="gene expression score to consider. Options: 'Z_SCORE', 'REGULATION' ")

    group.add_argument('--drug-based-batch-end', action = 'store_true',
                       help="if true, at each epoch once all the drug_drug batches are used for training, the epoch ends")



    ################### VARYING SETTINGS #####################################################
    group.add_argument('--cvdir', type=str, default="refactor",
                       help="folder to save cross validation folds ")

    group.add_argument('--save-model', action='store_true',
                       help="true means the trained model will be saved")
    #evaluation arguments
    group.add_argument('--force-cvdir', action = 'store_true')
    group.add_argument('--train', action = 'store_true')
    group.add_argument('--eval', action = 'store_false')
    ############################################################################################

    group.add_argument('--recall', type=float,
                       default=0.3,
                       help="recall value for early precision")

    return parser

def prepare_synergy_pairs(synergy_df,number_of_top_cell_lines,top_k_percent, apply_threshold=False ):
    ##***load drug-drug synergy dataset. This contains drugs for which we have atleast one target info before removing non-PPI targets
    ################based on top k percent###################
    cell_lines = synergy_df['Cell_line'].unique()
    drug_pairs_per_cell_line = {x: 0 for x in cell_lines}
    for row in synergy_df.itertuples():
        drug_pairs_per_cell_line[row.Cell_line] += 1
    drug_pairs_per_cell_line = dict(sorted(drug_pairs_per_cell_line.items(),\
                                           key=lambda item: item[1], reverse=True))


    top_k_cell_lines = list(drug_pairs_per_cell_line.keys())[0:number_of_top_cell_lines]

    synergy_df_new = pd.DataFrame()
    non_synergy_df = pd.DataFrame()
    cell_line_wise_threshold = {x:0 for x in top_k_cell_lines}
    cell_line_wise_drug_pairs = {x:0 for x in top_k_cell_lines}

    #keep only the top k cell lines
    for cell_line in top_k_cell_lines:
        cell_line_df_init = synergy_df[synergy_df['Cell_line'] == cell_line]
        cell_line_df = synergy_df[synergy_df['Cell_line']==cell_line]


        cell_line_df.sort_values(by='Loewe', inplace= True, ascending=False)
        cell_line_df.reset_index(drop = True, inplace= True)

        if not apply_threshold:
            number_of_pairs_in_top_k_percent = (top_k_percent/100.0) * len(cell_line_df)
            cell_line_df = cell_line_df.loc[0:number_of_pairs_in_top_k_percent]

            cell_line_wise_threshold[cell_line] = cell_line_df['Loewe'].min()
            cell_line_wise_drug_pairs[cell_line] = number_of_pairs_in_top_k_percent
        else:
            threshold_val = 0
            cell_line_df = cell_line_df[cell_line_df['Loewe']> threshold_val]

            cell_line_wise_threshold[cell_line] = 0
            cell_line_wise_drug_pairs[cell_line] = len(cell_line_df)


        non_syn_cell_line_df =cell_line_df_init[~cell_line_df_init.index.isin(cell_line_df.index)]
        non_synergy_df = pd.concat([non_synergy_df, non_syn_cell_line_df], axis=0, ignore_index=True)

        # hist_data = list(cell_line_df['Loewe'])
        # plt.hist(hist_data, weights=np.zeros_like(hist_data) + 1. / len(hist_data), bins=10)
        # plt.title(cell_line)
        # plt.show()
        # plt.clf()

        synergy_df_new = pd.concat([synergy_df_new, cell_line_df], axis = 0, ignore_index=True)
    synergy_df_new['Loewe_label'] =  pd.Series(np.ones(len(synergy_df_new)), dtype=int)
    non_synergy_df['Loewe_label'] =  pd.Series(np.zeros(len(non_synergy_df)), dtype=int)
    # print(synergy_df_new.head())

    print('cell line wise threshold: ',cell_line_wise_threshold.values())
    plt.scatter(cell_line_wise_threshold.keys(), cell_line_wise_threshold.values())
    plt.xticks(rotation='vertical', fontsize=9)
    plt.title('Cell line wise thresholds for being considered as synergistic')
    plt.tight_layout(pad=1, w_pad=0.4, h_pad=0.5)
    plt.show()
    plt.clf()

    plt.scatter(cell_line_wise_drug_pairs.keys(), cell_line_wise_drug_pairs.values(), vmin=0)
    plt.xticks(rotation='vertical', fontsize=9)
    plt.title('Cell line wise synergistic drug pairs')
    plt.xlabel('cell lines')
    plt.ylabel('number of drug pairs')
    plt.tight_layout(pad=1, w_pad=0.4, h_pad=0.5)

    plt.show()
    plt.clf()

    return synergy_df_new, non_synergy_df


def cell_line_2_tissue_map(config_map):
    cell_line_2_tissue_mapping_file = config_map['project_dir']+config_map['inputs']['mapping']
    cell_line_2_tissue_df = pd.read_csv(cell_line_2_tissue_mapping_file,sep='\t')
    cell_line_2_tissue_dict = dict(zip(cell_line_2_tissue_df['cellline'],cell_line_2_tissue_df['tissue']))
    return cell_line_2_tissue_dict

def preprocess_tissunet_ppi(config_map):
    project_dir = config_map['project_dir']
    ppi_network_dir = project_dir + config_map['inputs']['ppi']['tissuenet']
    ppi_dict = {}
    genes_in_ppi = set()
    for filename in os.listdir(ppi_network_dir):
        if 'uniprot' in filename:
            ppi_network_file = ppi_network_dir + filename
            tissue_name = filename.split('.')[0].replace('uniprot_','')

            ppi_df = pd.read_csv(ppi_network_file, sep='\t', index_col=None)
            ppi_dict[tissue_name] = ppi_df

            genes_in_ppi = genes_in_ppi.union(set(ppi_df['p1']).union(set(ppi_df['p2'])))

    gene_node_2_idx = dict(zip(list(genes_in_ppi), list(range(len(genes_in_ppi)))))

    return gene_node_2_idx, genes_in_ppi, ppi_dict

    
def preprocess_inputs(synergy_df, init_non_synergy_df, use_non_syn_df_in_preprocess, config_map):
    #outputs: ppi_sparse_matrix: a list of dataframes containing protein-protein interaction matrix from tissuenet
    # gene_node_2_idx (dictionray): PPI gene to index
    #the following three datastructures we do not have such a drug which is present in one but not in other two. The drugs are such that\
    #for the drug has at least one target in PPI, is present in synergy_df, has maccs_keys feature available.
    # drug_target_df:
    # drug_maccs_keys_feature_df:
    # synergy_df:
    project_dir = config_map['project_dir']

    drug_target_file = project_dir + config_map['inputs']['drug']['target']

    gene_node_2_idx, genes_in_ppi, ppi_dict = preprocess_tissunet_ppi(config_map)

    '''create drug-target network. This 'drug_target_map.tsv' '''
    drug_target_df_init = pd.read_csv(drug_target_file, sep='\t', index_col=0, header=0, dtype=str)


    genes_as_drug_target = list(set(drug_target_df_init['uniprot_id']))
    
    targets_not_in_ppi = list(set(genes_as_drug_target) - set(genes_in_ppi))

    print('\nCheck if any drug has been removed after excluding non PPI proteins:\n')

    drug_target_df = drug_target_df_init[~drug_target_df_init['uniprot_id'].isin(targets_not_in_ppi)]


    '''find common drugs between thresholded synergy_df and filtered drug_target_df'''
    if use_non_syn_df_in_preprocess:
        drugs_in_drugcombdb = set(synergy_df['Drug1_pubchem_cid']).union \
            (set(synergy_df['Drug2_pubchem_cid'])).union \
            (set(init_non_synergy_df['Drug1_pubchem_cid'])).union \
            (set(init_non_synergy_df['Drug2_pubchem_cid']))
    else:
        drugs_in_drugcombdb = set(synergy_df['Drug1_pubchem_cid']).union \
            (set(synergy_df['Drug2_pubchem_cid']))

    drugs_having_target_info = set(drug_target_df['pubchem_cid'])
    common = list(drugs_in_drugcombdb.intersection(drugs_having_target_info))
    print('total drugcombdb drug: ', len(drugs_in_drugcombdb))
    print('target info present for: ', len(drugs_having_target_info))
    print('common: ',len(common))
    
    
    '''keep only the common drugs in synergy_df and drug_target_df'''
    drug_target_df = drug_target_df[drug_target_df['pubchem_cid'].isin(common)]
    synergy_df = synergy_df[synergy_df['Drug1_pubchem_cid'].isin(common) & synergy_df['Drug2_pubchem_cid'].isin(common)]
    init_non_synergy_df = init_non_synergy_df[init_non_synergy_df['Drug1_pubchem_cid'].isin(common) & \
                                              init_non_synergy_df['Drug2_pubchem_cid'].isin(common)]

    ''' add index column to synergy_df synergy_df.set_index(pd.Series(range(len(synergy_df))), inplace=True)'''
    synergy_df.reset_index(inplace=True)
    init_non_synergy_df.reset_index(inplace=True)
    # print('final number of cell lines:', len(synergy_df['Cell_line'].unique()))

    return ppi_dict, gene_node_2_idx, drug_target_df, synergy_df, init_non_synergy_df
    
        
def preprocess_drug_feat(synergy_df, init_non_synergy_df, use_non_syn_df_in_preprocess, config_map):
    # drug_feature ready
    # keep only features for the drugs which have target info, macc_keys info, synergy_info and whose targets(at least 1) are in PPI.\
    # In short, who are in synergy_df

    if use_non_syn_df_in_preprocess:
        drugs_in_drugcombdb = set(synergy_df['Drug1_pubchem_cid']).union \
            (set(synergy_df['Drug2_pubchem_cid'])).union \
            (set(init_non_synergy_df['Drug1_pubchem_cid'])).union \
            (set(init_non_synergy_df['Drug2_pubchem_cid']))

    else:
        drugs_in_drugcombdb = set(synergy_df['Drug1_pubchem_cid']).union \
            (set(synergy_df['Drug2_pubchem_cid']))



    drug_maccs_keys_file = config_map['project_dir'] + config_map['inputs']['drug']['maccs_keys']
    drug_maccs_keys_feature_df = pd.read_csv(drug_maccs_keys_file, sep = '\t', index_col=None, dtype={'pubchem_cid':str})
    drug_maccs_keys_feature_df = drug_maccs_keys_feature_df[drug_maccs_keys_feature_df['pubchem_cid'].isin(drugs_in_drugcombdb)]
    # print('common:' , (common), '\nmaccs key:',(list(drug_maccs_keys_feature_df['pubchem_cid'])))

    drug_maccs_keys_targets_file = config_map['project_dir'] + config_map['inputs']['drug']['maccs_keys_targets']
    drug_maccs_keys_targets_feature_df = pd.read_csv(drug_maccs_keys_targets_file, sep='\t', index_col=None,\
                                                     dtype={'pubchem_cid': str})
    drug_maccs_keys_targets_feature_df = drug_maccs_keys_targets_feature_df[drug_maccs_keys_targets_feature_df['pubchem_cid'].\
        isin(drugs_in_drugcombdb)]

    return drug_maccs_keys_feature_df, drug_maccs_keys_targets_feature_df


def main(config_map, **kwargs):

    print(kwargs.get('force_cvdir'), kwargs.get('train'), kwargs.get('eval'))
    force_cvdir = kwargs.get('force_cvdir')
    # print(force_run)


    #synergy data settings
    min_pairs_per_cell_line = config_map['synergy_data_settings']['min_pairs']
    max_pairs_per_cell_line =  config_map['synergy_data_settings']['max_pairs']
    threshold = config_map['synergy_data_settings']['threshold']['val']
    number_of_top_cell_lines = config_map['synergy_data_settings']['number_of_top_cell_lines']
    top_k_percent = config_map['synergy_data_settings']['top_k_percent_pairs']

    #cell line specific gene expression data settings
    exp_score = kwargs.get('exp_score')
    init_gene_expression_file_path =  config_map['project_dir']+config_map['inputs']['cell_lines'][exp_score]
    compressed_gene_expression_dir =  os.path.dirname(init_gene_expression_file_path)
    # num_compressed_gene = config_map['autoencoder_settings']['num_compressed_gene']

    # cross val settings
    # test_frac = config_map['ml_models_settings']['cross_val']['test_frac']
    val_frac = config_map['ml_models_settings']['cross_val']['val_frac']
    cross_val_types = config_map['ml_models_settings']['cross_val']['types']
    number_of_folds = config_map['ml_models_settings']['cross_val']['folds']
    neg_fact = config_map['ml_models_settings']['cross_val']['neg_fact']
    neg_sampling_type = kwargs.get('sampling')
    apply_threshold = kwargs.get('apply_threshold')

    number_of_runs = config_map['ml_models_settings']['runs']
    algs = config_map['ml_models_settings']['algs']
    project_dir = config_map['project_dir']
    synergy_file = project_dir + config_map['inputs']['synergy']

    cell_line_2_tissue_dict = cell_line_2_tissue_map(config_map)
    #get only top k percent of top cell lines
    synergy_df = pd.read_csv(synergy_file, sep='\t', dtype={'Drug1_pubchem_cid': str,
                                                            'Drug2_pubchem_cid': str,
                                                            'Cell_line': str,
                                                            'Loewe': np.float64,
                                                            'Bliss': np.float64,
                                                            'ZIP': np.float64})

    # only keep the cell_lines for which we have tissue mapping and thus PPI network available
    synergy_df = synergy_df[synergy_df['Cell_line'].isin(list(cell_line_2_tissue_dict.keys()))]

    #then keep only top percent/thrsholded pairs of number_of_top_cell_lines
    #synergy_df and non_synergy_df have the same cell lines in them
    synergy_df, init_non_synergy_df = prepare_synergy_pairs(synergy_df, number_of_top_cell_lines, top_k_percent, apply_threshold)

    #
    # #after the following operation both synergy_df and gene_expression_feature_df will have same cell lines in them
    # gene_expression_feature_df = gene_expression_feature_df[gene_expression_feature_df['cell_line_name'].\
    #                             isin(list(synergy_df['Cell_line']))]

    #now map cell line names to index for future use in different models
    cell_line_2_idx, idx_2_cell_line = utils.generate_cell_line_idx_mapping(synergy_df)


    #procesing network data
    if neg_sampling_type=='no':
        use_non_syn_df_in_preprocess=True
    else:
        use_non_syn_df_in_preprocess = False

    ppi_dict, gene_node_2_idx, drug_target_df, synergy_df, init_non_synergy_df = \
        preprocess_inputs(synergy_df, init_non_synergy_df, use_non_syn_df_in_preprocess, config_map)

    print('final drug pairs per cell line', synergy_df.groupby('Cell_line').count())

    # processing durg feature data
    drug_maccs_keys_feature_df, drug_maccs_keys_targets_feature_df = preprocess_drug_feat(synergy_df,\
                                            init_non_synergy_df, use_non_syn_df_in_preprocess, config_map)

    should_run_algs = []
    for alg in algs:
        if algs[alg]['should_run'] == True:
            should_run_algs.append(alg)
    print('should_run_algs: ', should_run_algs)

    # dict of dict to hold different types of cross val split folds
    type_wise_pos_cross_folds = {cross_val_type: dict() for cross_val_type in cross_val_types}
    type_wise_neg_cross_folds = {cross_val_type: dict() for cross_val_type in cross_val_types}

    # generate model prediction and result
    if kwargs.get('train') == True:
        for run_ in range(number_of_runs):
            print("RUN NO:", run_)

            out_params = prepare_output_prefix(split_type, config_map, **kwargs) + '/run_' + str(run_) + '/'
            cross_val_dir = config_map['project_dir'] + config_map['output_dir']['split'] + out_params

            pos_train_test_val_file = cross_val_dir + 'pos_train_test_val.pkl'
            neg_train_test_val_file = cross_val_dir + 'neg_train_test_val.pkl'
            non_syn_file = cross_val_dir + 'non_synergy.tsv'
            syn_file = cross_val_dir + 'synergy.tsv'
            if (not os.path.exists(pos_train_test_val_file)) | (not os.path.exists(neg_train_test_val_file)) | \
                    (not os.path.exists(non_syn_file)) | (force_cvdir == True):
                # only cross validation splits

                pos_folds, neg_folds, \
                non_synergy_df = cross_val.create_test_val_train_cross_val_folds \
                    (synergy_df, init_non_synergy_df, split_type, number_of_folds, neg_fact, val_frac,
                     neg_sampling_type,
                     number_of_test_cell_lines)

                print('non_syn type: ', type(non_synergy_df))

                os.makedirs(os.path.dirname(pos_train_test_val_file), exist_ok=True)

                # pkl dump the train_test_val pos and neg fold

                with open(pos_train_test_val_file, 'wb') as handle:
                    pickle.dump(pos_folds, handle, protocol=pickle.HIGHEST_PROTOCOL)
                with open(neg_train_test_val_file, 'wb') as handle:
                    pickle.dump(neg_folds, handle, protocol=pickle.HIGHEST_PROTOCOL)
                non_synergy_df.to_csv(non_syn_file, index=True, sep='\t')
                synergy_df.to_csv(syn_file, index=True, sep='\t')

            non_synergy_df = pd.read_csv(non_syn_file, sep='\t', index_col=0, dtype={'Drug1_pubchem_cid': str, \
                                                                                     'Drug2_pubchem_cid': str,
                                                                                     'Cell_line': str,
                                                                                     'Loewe_label': int})
            with open(pos_train_test_val_file, 'rb') as handle:
                pos_folds = pickle.load(handle)
            with open(neg_train_test_val_file, 'rb') as handle:
                neg_folds = pickle.load(handle)

            # print('final number of drug pairs going into training: ', len(synergy_df))
            for alg in should_run_algs:
                result_dir = config_map['project_dir'] + config_map['output_dir']['result'] + alg + '/' + out_params
                os.makedirs(result_dir, exist_ok=True)

                params_list = prepare_alg_param_list(alg, config_map)
                # print('final number of drug pairs going into training: ', len(synergy_df))
                for alg in should_run_algs:
                    out_dir = config_map['project_dir'] + config_map['output_dir'] + alg + '/' + cross_val_type + '/' + \
                                    'pairs_' + str(min_pairs_per_cell_line) + '_' + \
                                    str(max_pairs_per_cell_line) + '_th_' + str(threshold) + '_cell_lines_' + str(number_of_top_cell_lines) + \
                                    '_percent_' + str(top_k_percent) + \
                                    '_' + 'neg_' + str(neg_fact) + '_neg_sampling_' + neg_sampling_type + '_val_frac_'+str(val_frac)+'_'+ kwargs.get('cvdir')+'/'+'run_' + str(run_)+'/'

                    os.makedirs(out_dir, exist_ok=True)

                    if alg=='synverse_tissuenet':
                        print('Model running: ', alg)

                        synverse_params_list = prepare_synverse_tissuenet_param_settings(config_map)

                        for synverse_params in synverse_params_list:
                            print(' synverse_params :',synverse_params)
                            synverse_tissuenet.run_synverse_model(copy.deepcopy(ppi_dict), gene_node_2_idx, drug_target_df,
                                                        drug_maccs_keys_feature_df,
                                                        synergy_df, non_synergy_df,
                                                        cell_line_2_idx,cell_line_2_tissue_dict, idx_2_cell_line,
                                                        type_wise_pos_cross_folds[cross_val_type],
                                                        type_wise_neg_cross_folds[cross_val_type],
                                                        cross_val_dir, neg_sampling_type,
                                                        out_dir,synverse_params,  config_map,
                                                        use_drug_based_batch_end=kwargs.get('drug_based_batch_end'))

if __name__ == "__main__":
    config_map, kwargs = parse_args()
    main(config_map, **kwargs)

