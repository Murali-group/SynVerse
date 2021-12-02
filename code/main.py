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
from scipy.io import loadmat
import utils
from utils import *
import pickle
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
#import subprocess
# sys.path.insert(0, '/home/tasnina/Projects/Synverse/')
import models.gnn.run_synverse as synverse
# import models.gnn.run_synverse_nogenex as synverse_nogenex
# import models.gnn.run_synverse_v2 as synverse_v2
# import models.gnn.run_synverse_v3 as synverse_v3
# import models.gnn.run_synverse_v4 as synverse_v4


# import models.gnn.run_decagon as decagon
# import models.deepsynergy as deepsynergy

from data_split import cross_validation as cross_val

import evaluation.evaluation_handler as evaluation_handler

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


def preprocess_inputs(string_cutoff,
                      synergy_df, init_non_synergy_df, use_non_syn_df_in_preprocess, config_map):
    #outputs: ppi_sparse_matrix: protein-protein interaction matrix from STRING with cutoff at 700
    # gene_node_2_idx (dictionray): PPI gene to index
    #the following three datastructures we do not have such a drug which is present in one but not in other two. The drugs are such that\
    #for the drug has at least one target in PPI, is present in synergy_df, has maccs_keys feature available.
    # drug_target_df:
    # drug_maccs_keys_feature_df:
    # synergy_df:
    project_dir = config_map['project_dir']

    drug_target_file = project_dir + config_map['inputs']['drug']['target']

    
    # mat: /c"+str(string_cutoff)+"_combined_score_sparse_net.mat"
        # nodes: "inputs/networks/c"+str(string_cutoff)+"_combined_score_node_ids.txt"
    ppi_network_file = project_dir + config_map['inputs']['ppi']['string']+ "c" + str(string_cutoff)+"_combined_score_sparse_net.mat"
    
    # ppi_network_file = '/home/tasnina/Projects/SynVerse/inputs/networks/c700_combined_score_sparse_net.mat'
    
    ppi_node_to_idx_file = project_dir +config_map['inputs']['ppi']['string']+ "c" + str(string_cutoff)+"_combined_score_node_ids.txt"

    ppi_sparse_matrix = loadmat(ppi_network_file)['Networks'][0][0]

    print(ppi_sparse_matrix.shape)

    gene_node_2_idx = pd.read_csv(ppi_node_to_idx_file, sep='\t', header=None, names=['gene','index'])
    gene_node_2_idx= dict(zip(gene_node_2_idx['gene'], gene_node_2_idx['index']))

    # synergy_df = prepare_synergy_pairs(synergy_file, number_of_top_cell_lines,top_k_percent)

    
    synergistics_drugs = set(list(synergy_df['Drug1_pubchem_cid'])).union(set(list(synergy_df['Drug2_pubchem_cid'])))
    # print('number of drugs after applying threshold on synergy data:' , len(synergistics_drugs))

    ##***create drug-target network. This 'drug_target_map.tsv'
    drug_target_df_init = pd.read_csv(drug_target_file, sep ='\t', index_col = 0, header = 0, dtype = str)
    
    #mapping genes to their index as in ppi
    genes_in_ppi = gene_node_2_idx.keys()
    genes_as_drug_target = list(drug_target_df_init['uniprot_id'])
    
    targets_not_in_ppi = list(set(genes_as_drug_target)- set(genes_in_ppi))

    
    print('\nCheck if any drug has been removed after excluding non PPI proteins:\n')

    #for now remove the 105 targets which do not have ppi data, from drug target list
    #this will also remove any drug that has no target remianing after removing the non PPI targets.
    drug_target_df = drug_target_df_init[~drug_target_df_init['uniprot_id'].isin(targets_not_in_ppi)]


    #find common drugs between thresholded synergy_df and filtered drug_target_df
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
    
    
    #keep only the common drugs in synergy_df and drug_target_df
    drug_target_df = drug_target_df[drug_target_df['pubchem_cid'].isin(common)]
    synergy_df = synergy_df[synergy_df['Drug1_pubchem_cid'].isin(common) & synergy_df['Drug2_pubchem_cid'].isin(common)]
    init_non_synergy_df = init_non_synergy_df[init_non_synergy_df['Drug1_pubchem_cid'].isin(common) & \
                                              init_non_synergy_df['Drug2_pubchem_cid'].isin(common)]

    # add index column to synergy_df
    # synergy_df.set_index(pd.Series(range(len(synergy_df))), inplace=True)
    synergy_df.reset_index(inplace=True)
    init_non_synergy_df.reset_index(inplace=True)
    # print('final number of cell lines:', len(synergy_df['Cell_line'].unique()))

    return ppi_sparse_matrix, gene_node_2_idx, drug_target_df, synergy_df, init_non_synergy_df
    
        
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
    string_cutoff = config_map['string_network_preparation_settings']['string_cutoff']
    number_of_top_cell_lines = config_map['synergy_data_settings']['n_top_cell_lines']
    number_of_test_cell_lines = config_map['synergy_data_settings']['n_rare_cell_lines']
    top_k_percent = config_map['synergy_data_settings']['percent']

    #cell line specific gene expression data settings
    exp_score = kwargs.get('exp_score')
    init_gene_expression_file_path =  config_map['project_dir']+config_map['inputs']['cell_lines'][exp_score]


    # cross val settings
    val_frac = config_map['split']['val_frac']
    split_type = config_map['split']['type']
    number_of_folds = config_map['split']['folds']
    neg_fact = config_map['split']['neg_frac']
    neg_sampling_type = config_map['split']['sampling']
    apply_threshold = config_map['synergy_data_settings']['apply_threshold']

    number_of_runs = config_map['runs']
    algs = config_map['ml_models_settings']['algs']
    project_dir = config_map['project_dir']
    synergy_file = project_dir + config_map['inputs']['synergy']


    #get only top k percent of top cell lines
    synergy_df = pd.read_csv(synergy_file, sep='\t', dtype={'Drug1_pubchem_cid': str,
                                                            'Drug2_pubchem_cid': str,
                                                            'Cell_line': str,
                                                            'Loewe': np.float64,
                                                            'Bliss': np.float64,
                                                            'ZIP': np.float64})
    #preprocess gene expression data
    if config_map['genex_data_settings']['reduce_dim']:
        gene_expression_file = config_map['project_dir'] + config_map['inputs']['cell_lines']['compressed_gene_expression']
    else:
        gene_expression_file = config_map['project_dir'] + config_map['inputs']['cell_lines']['uncompressed_gene_expression']
    gene_expression_feature_df = pd.read_csv(gene_expression_file, sep='\t', index_col=0)

    # only keep the cell_lines for which we have gene expression value available
    synergy_df = synergy_df[synergy_df['Cell_line'].isin(list(gene_expression_feature_df['cell_line_name']))]

    #then keep only top percent/thrsholded pairs of number_of_top_cell_lines
    #synergy_df and non_synergy_df have the same cell lines in them
    synergy_df, init_non_synergy_df = prepare_synergy_pairs(synergy_df, number_of_top_cell_lines, top_k_percent, \
                                                            apply_threshold)


    #after the following operation both synergy_df and gene_expression_feature_df will have same cell lines in them
    gene_expression_feature_df = gene_expression_feature_df[gene_expression_feature_df['cell_line_name'].\
                                isin(list(synergy_df['Cell_line']))]

    #now map cell line names to index for future use in different models
    cell_line_2_idx, idx_2_cell_line = utils.generate_cell_line_idx_mapping(synergy_df)
    gene_expression_feature_df['cell_line_index'] = gene_expression_feature_df['cell_line_name'].\
                                                    apply(lambda x: cell_line_2_idx[x])
    gene_expression_feature_df.drop('cell_line_name', axis=1, inplace=True)
    gene_expression_feature_df.set_index('cell_line_index', inplace=True, drop=True)



    if neg_sampling_type=='no':
        use_non_syn_df_in_preprocess=True
    else:
        use_non_syn_df_in_preprocess = False

    # procesing network data
    ppi_sparse_matrix, gene_node_2_idx, drug_target_df, synergy_df, init_non_synergy_df = \
        preprocess_inputs(string_cutoff, synergy_df,\
                          init_non_synergy_df, use_non_syn_df_in_preprocess, config_map)

    print('final drug pairs per cell line', synergy_df.groupby('Cell_line').count())

    # processing durg feature data
    drug_maccs_keys_feature_df, drug_maccs_keys_targets_feature_df = preprocess_drug_feat(synergy_df,\
                                            init_non_synergy_df,use_non_syn_df_in_preprocess,  config_map)


    should_run_algs = []
    for alg in algs:
        if algs[alg]['should_run'] == True:
            should_run_algs.append(alg)
    print('should_run_algs: ', should_run_algs)

    # generate model prediction and result
    if kwargs.get('train')==True:
        for run_ in range(number_of_runs):
            print("RUN NO:", run_)

            out_params = prepare_output_prefix(split_type, config_map, **kwargs) + '/run_' + str(run_) + '/'
            cross_val_dir = config_map['project_dir'] + config_map['output_dir']['split'] + out_params

            pos_train_test_val_file = cross_val_dir + 'pos_train_test_val.pkl'
            neg_train_test_val_file = cross_val_dir + 'neg_train_test_val.pkl'
            non_syn_file = cross_val_dir + 'non_synergy.tsv'
            syn_file = cross_val_dir + 'synergy.tsv'
            if (not os.path.exists(pos_train_test_val_file))|(not os.path.exists(neg_train_test_val_file))|\
                    (not os.path.exists(non_syn_file))|(force_cvdir == True):

                # only cross validation splits

                pos_folds, neg_folds,\
                non_synergy_df = cross_val.create_test_val_train_cross_val_folds\
                    (synergy_df,init_non_synergy_df, split_type, number_of_folds, neg_fact, val_frac, neg_sampling_type,
                     number_of_test_cell_lines)

                print('non_syn type: ',type(non_synergy_df))

                os.makedirs(os.path.dirname(pos_train_test_val_file), exist_ok=True)

                #pkl dump the train_test_val pos and neg fold

                with open(pos_train_test_val_file, 'wb') as handle:
                    pickle.dump(pos_folds, handle, protocol=pickle.HIGHEST_PROTOCOL)
                with open(neg_train_test_val_file, 'wb') as handle:
                    pickle.dump(neg_folds, handle,protocol=pickle.HIGHEST_PROTOCOL)
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

                if alg in ['synverse', 'synverse_v4','decagon','synverse_nogenex']:
                    print('Model running: ', alg)
                    for synverse_params in params_list:
                        print(' synverse_params :',synverse_params)
                        synverse.run_synverse_model(alg, ppi_sparse_matrix, gene_node_2_idx, drug_target_df,
                                                    drug_maccs_keys_feature_df,
                                                    synergy_df, non_synergy_df,
                                                    cell_line_2_idx, idx_2_cell_line,
                                                    pos_folds,
                                                    neg_folds,
                                                    cross_val_dir, neg_sampling_type,
                                                    result_dir,synverse_params,  config_map,
                                                    gene_expression_feature_df)


                if alg == 'deepsynergy':
                    print('Model running: ', alg)

                    for ds_params in params_list:
                        deepsynergy.run_deepsynergy_model(copy.deepcopy(drug_maccs_keys_targets_feature_df),
                                                          gene_expression_feature_df,
                                                          synergy_df, non_synergy_df, cell_line_2_idx, idx_2_cell_line,
                                                          pos_folds,
                                                          neg_folds,
                                                          ds_params, result_dir, config_map)


                if alg=='svm':
                    print('Model running: ', alg)
                    use_genex = kwargs.get('use_genex')
                    use_target = kwargs.get('use_target')

                    for svm_params in params_list:
                        svm.run_svm_model(drug_maccs_keys_targets_feature_df, drug_maccs_keys_feature_df,
                                                          gene_expression_feature_df,use_genex,use_target,
                                                          synergy_df, non_synergy_df, cell_line_2_idx, idx_2_cell_line,
                                                          pos_folds,
                                                          neg_folds,
                                                          svm_params, result_dir, config_map)

                if alg=='gbr':
                    print('Model running: ', alg)
                    use_genex = kwargs.get('use_genex')
                    use_target = kwargs.get('use_target')

                    for gbr_params in params_list:
                        gbr.run_gbr_model(drug_maccs_keys_targets_feature_df, drug_maccs_keys_feature_df,
                                                          gene_expression_feature_df,use_genex,use_target,
                                                          synergy_df, non_synergy_df, cell_line_2_idx, idx_2_cell_line,
                                                          pos_folds,
                                                          neg_folds,
                                                          gbr_params, result_dir, config_map)



    ################### PLOT ######################################
    if kwargs.get('eval') == True:
        param_settings_dict = {alg: [] for alg in should_run_algs} #this will contain the hyperparam and model param options considered for each alg
        for alg in should_run_algs:
                param_settings_dict[alg] = prepare_alg_param_list(alg, config_map)
        # evaluation_handler.evaluate(should_run_algs, param_settings_dict, cross_val_type, kwargs, config_map)
        # evaluation_handler.find_best_param(should_run_algs, param_settings_dict, cross_val_type, kwargs, config_map)
        out_params = prepare_output_prefix(split_type, config_map, **kwargs)
        evaluation_handler.plot_best_models(should_run_algs, param_settings_dict, out_params, kwargs, config_map)
if __name__ == "__main__":
    config_map, kwargs = parse_args()
    main(config_map, **kwargs)

