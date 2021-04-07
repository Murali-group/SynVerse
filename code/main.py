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
import numpy as np
#from scipy import sparse
from scipy.io import loadmat
import pandas as pd
import os
import  time
#import subprocess
# sys.path.insert(0, '/home/tasnina/Projects/Synverse/')
import cross_validation as cross_val
# import models.synverse.run_synverse as synverse
#
# import models.deepsynergy as deepsynergy
#
import models.decagon_handler.run_decagon as decagon
import evaluation.performance_metric_plot as metric_plot



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
    group.add_argument('--config', type=str, default="/home/tasnina/Projects/SynVerse/code/config-files/master-config.yaml",
                       help="Configuration file for this script.")
    group.add_argument('--synergy', type=str, default="/synergy/synergy_labels.tsv",
                       help="Configuration file for this script.")
    group.add_argument('--drug_target', type=str, default="/drugs/drug_target_map.tsv",
                       help="Configuration file for this script.")

    group = parser.add_argument_group('FastSinkSource Pipeline Options')
    group.add_argument('--alg', '-A', dest='algs', type=str, action="append",
                       help="Algorithms for which to get results. Must be in the config file.\
                           If not specified, will get the list of algs with should_run set to True in the config file")
    group.add_argument('--force-run', action='store_true', default=False)

    return parser

def preprocess_inputs(string_cutoff,minimum_number_of_synergistic_pairs_per_cell_line,maximum_number_of_synergistic_pairs_per_cell_line,config_map):
    #outputs: ppi_sparse_matrix: protein-protein interaction matrix from STRING with cutoff at 700
    # gene_node_2_idx (dictionray): PPI gene to index
    #the following three datastructures we do not have such a drug which is present in one but not in other two. The drugs are such that\
    #for the drug has at least one target in PPI, is present in synergy_df, has maccs_keys feature available.
    # drug_target_df:
    # drug_maccs_keys_feature_df:
    # synergy_df:
    project_dir = config_map['project_dir']
    synergy_file = project_dir + config_map['inputs']['synergy']
    drug_target_file = project_dir + config_map['inputs']['drug']['target']
    
    # mat: /c"+str(string_cutoff)+"_combined_score_sparse_net.mat"
        # nodes: "inputs/networks/c"+str(string_cutoff)+"_combined_score_node_ids.txt"
    ppi_network_file = project_dir + config_map['inputs']['ppi']+ "c" + str(string_cutoff)+"_combined_score_sparse_net.mat"
    
    # ppi_network_file = '/home/tasnina/Projects/SynVerse/inputs/networks/c700_combined_score_sparse_net.mat'
    
    ppi_node_to_idx_file = project_dir +config_map['inputs']['ppi']+ "c" + str(string_cutoff)+"_combined_score_node_ids.txt"


    ppi_sparse_matrix = loadmat(ppi_network_file)['Networks'][0][0]

    print(ppi_sparse_matrix.shape)

    gene_node_2_idx = pd.read_csv(ppi_node_to_idx_file, sep='\t', header=None, names=['gene','index'])
    gene_node_2_idx= dict(zip(gene_node_2_idx['gene'], gene_node_2_idx['index']))
    
    
    
    
    ##***load drug-drug synergy dataset. This contains drugs for which we have atleast one target info before removing non-PPI targets

    synergy_df = pd.read_csv(synergy_file, sep = '\t',dtype={'Drug1_pubchem_cid':str,\
                'Drug2_pubchem_cid': str, 'Cell_line': str, 'Loewe': np.float64,\
                'Bliss': np.float64, 'ZIP': np.float64})


    #threshold on Loewe score to determine if two drugs are synergistic or not in synergy_df keep only the pairs who passed the threshold

    if(config_map['synergy_data_settings']['threshold']['should_apply']==True):
        threshold = config_map['synergy_data_settings']['threshold']['val']
        synergy_df['Loewe_label'] = synergy_df['Loewe'].astype(np.float64).apply(lambda x: 0 if x<threshold else 1)
        synergy_df = synergy_df[synergy_df['Loewe_label']==1]



    cell_lines  = synergy_df['Cell_line'].unique()
    drug_pairs_per_cell_line = {x : 0 for x in cell_lines}
    
    for row in synergy_df.itertuples():
            drug_pairs_per_cell_line[row.Cell_line] += 1
    # print('drug pairs per cell line', drug_pairs_per_cell_line )
    print('no of cell lines before: ', len(synergy_df['Cell_line'].unique()))
    
    for cell_line in cell_lines:
        if (drug_pairs_per_cell_line[cell_line]<minimum_number_of_synergistic_pairs_per_cell_line) |\
                (drug_pairs_per_cell_line[cell_line]>maximum_number_of_synergistic_pairs_per_cell_line):
            synergy_df = synergy_df[synergy_df['Cell_line']!=cell_line]
    
    print('no of cell lines after: ', len(synergy_df['Cell_line'].unique()))
    
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
    drugs_in_drugcombdb = set(synergy_df['Drug1_pubchem_cid']).union\
                                (set(synergy_df['Drug2_pubchem_cid']))
    drugs_having_target_info = set(drug_target_df['pubchem_cid'])
    common = list(drugs_in_drugcombdb.intersection(drugs_having_target_info))
    print('total drugcombdb drug: ', len(drugs_in_drugcombdb))
    print('target info present for: ', len(drugs_having_target_info))
    print('common: ',len(common))
    
    
    #keep only the common drugs in synergy_df and drug_target_df
    drug_target_df = drug_target_df[drug_target_df['pubchem_cid'].isin(common)]
    synergy_df = synergy_df[synergy_df['Drug1_pubchem_cid'].isin(common) & synergy_df['Drug2_pubchem_cid'].isin(common)]

    # add index column to synergy_df
    # synergy_df.set_index(pd.Series(range(len(synergy_df))), inplace=True)
    synergy_df.reset_index(inplace=True)

    # print('final number of cell lines:', len(synergy_df['Cell_line'].unique()))

    return ppi_sparse_matrix, gene_node_2_idx, drug_target_df, synergy_df
    
        
def drug_feat_preprocess(synergy_df, config_map):
    # drug_feature ready
    # keep only features for the drugs which have target info, macc_keys info, synergy_info and whose targets(at least 1) are in PPI.\
    # In short, who are in synergy_df

    drugs_in_synergy_df = set(synergy_df['Drug1_pubchem_cid']).union \
        (set(synergy_df['Drug2_pubchem_cid']))

    drug_maccs_keys_file = config_map['project_dir'] + config_map['inputs']['drug']['maccs_keys']
    drug_maccs_keys_feature_df = pd.read_csv(drug_maccs_keys_file, sep = '\t', index_col=None, dtype={'pubchem_cid':str})
    drug_maccs_keys_feature_df = drug_maccs_keys_feature_df[drug_maccs_keys_feature_df['pubchem_cid'].isin(drugs_in_synergy_df)]
    # print('common:' , (common), '\nmaccs key:',(list(drug_maccs_keys_feature_df['pubchem_cid'])))

    drug_maccs_keys_targets_file = config_map['project_dir'] + config_map['inputs']['drug']['maccs_keys_targets']
    drug_maccs_keys_targets_feature_df = pd.read_csv(drug_maccs_keys_targets_file, sep='\t', index_col=None, dtype={'pubchem_cid': str})
    drug_maccs_keys_targets_feature_df = drug_maccs_keys_targets_feature_df[drug_maccs_keys_targets_feature_df['pubchem_cid'].\
        isin(drugs_in_synergy_df)]

    return drug_maccs_keys_feature_df, drug_maccs_keys_targets_feature_df



def main(config_map, **kwargs):
    string_cutoff = config_map['string_network_preparation_settings']['string_cutoff']
    min_pairs_per_cell_line = config_map['synergy_data_settings']['min_pairs']
    max_pairs_per_cell_line =  config_map['synergy_data_settings']['max_pairs']
    threshold = config_map['synergy_data_settings']['threshold']['val']

    ppi_sparse_matrix, gene_node_2_idx, drug_target_df, synergy_df = \
        preprocess_inputs(string_cutoff,min_pairs_per_cell_line,max_pairs_per_cell_line,config_map)

    drug_maccs_keys_feature_df,drug_maccs_keys_targets_feature_df = drug_feat_preprocess(synergy_df, config_map)

    cross_val_types = config_map['ml_models_settings']['cross_val']['types']
    number_of_folds = config_map['ml_models_settings']['cross_val']['folds']
    neg_fact = config_map['ml_models_settings']['cross_val']['neg_fact']

    number_of_runs = config_map['ml_models_settings']['cross_val']['runs']
    test_frac = config_map['ml_models_settings']['cross_val']['test_frac']

    algs = config_map['ml_models_settings']['algs']


    should_run_algs = []
    for alg in algs:
        if algs[alg]['should_run'] == True:
            should_run_algs.append(alg)
    print(should_run_algs)



    # #generate model prediction and result
    for i in range(number_of_runs):
        #train-test split
        # train_synergy_df, test_synergy_df = cross_val.train_test_split(synergy_df, test_frac)
        # dict of dict to hold different types of cross val split folds
        type_wise_pos_cross_val_folds = {cross_val_type: dict() for cross_val_type in cross_val_types }
        type_wise_neg_cross_val_folds = {cross_val_type: dict() for cross_val_type in cross_val_types}
        for cross_val_type in cross_val_types:
            type_wise_pos_cross_val_folds[cross_val_type], type_wise_neg_cross_val_folds[cross_val_type], non_synergy_df = cross_val.\
                create_cross_val_folds(synergy_df, cross_val_type, number_of_folds,neg_fact)

            #save the output from cross-val
            cross_val_dir = config_map['project_dir']+ config_map['output_dir'] + 'cross_val/'+'/run_'+str(i)+ '_'+cross_val_type+'/pairs_' + str(min_pairs_per_cell_line)+'_'+\
                str(max_pairs_per_cell_line) + '_th_'+str(threshold)+'_'+'neg_'+str(neg_fact)+'_'+str(time.time())+'/'
            pos_fold_file = cross_val_dir + 'pos_folds.tsv'
            neg_fold_file = cross_val_dir + 'neg_folds.tsv'
            syn_df_file = cross_val_dir + 'synergy.tsv'
            non_syn_df_file = cross_val_dir +'non_synergy.tsv'

            os.makedirs(os.path.dirname(pos_fold_file), exist_ok=True)
            pd.DataFrame.from_dict(type_wise_pos_cross_val_folds[cross_val_type],orient='index').T.to_csv(pos_fold_file, sep='\t')
            pd.DataFrame.from_dict(type_wise_neg_cross_val_folds[cross_val_type],orient='index').T.to_csv(neg_fold_file, sep='\t')
            synergy_df.to_csv(syn_df_file, sep='\t')
            non_synergy_df.to_csv(non_syn_df_file,sep='\t')


        for alg in should_run_algs:
            out_dir = config_map['project_dir'] + config_map['output_dir'] + alg + '/' + \
                      'pairs_' + str(min_pairs_per_cell_line) + '_' + str(max_pairs_per_cell_line)+'_th_'+str(threshold)+'_'+'neg_'+str(neg_fact)+'/'
            if alg=='synverse':
                    print('Model running: ', alg)
                    synverse.run_synverse_model(ppi_sparse_matrix, gene_node_2_idx, drug_target_df, drug_maccs_keys_feature_df,
                               synergy_df, non_synergy_df, \
                               type_wise_pos_cross_val_folds['random'], type_wise_neg_cross_val_folds['random'], \
                               i, out_dir, config_map)

            if alg=='decagon':
                    print('Model running: ', alg)
                    decagon.run_decagon_model(ppi_sparse_matrix,gene_node_2_idx,\
                                                                      drug_target_df, drug_maccs_keys_feature_df,\
                        synergy_df, non_synergy_df, type_wise_pos_cross_val_folds['random'], type_wise_neg_cross_val_folds['random'],\
                                                                      i,out_dir, config_map)

            if alg=='deepsynergy':
                    print('Model running: ', alg)
                    # deepsynergy.run_deepsynergy_model(drug_maccs_keys_targets_feature_df,\
                    #     train_synergy_df, test_synergy_df, non_synergy_df, type_wise_pos_cross_val_folds['random'],type_wise_neg_cross_val_folds['random'], i,out_dir, config_map)
                    deepsynergy.run_deepsynergy_model(drug_maccs_keys_targets_feature_df, \
                                                      synergy_df, non_synergy_df,
                                                      type_wise_pos_cross_val_folds['random'],
                                                      type_wise_neg_cross_val_folds['random'], i, out_dir, config_map)

    ###################### PLOT ######################################
    # # read and plot result from already existing files
    ## pos_df_all_runs={alg:[] for alg in should_run_algs}
    ## neg_df_all_runs={alg:[] for alg in should_run_algs}

    # outputs_df = {alg: [] for alg in should_run_algs}
    # for alg in should_run_algs:
    #     for run_ in range(number_of_runs):
    #         out_dir = config_map['project_dir'] + config_map['output_dir'] + alg + '/'+\
    #          'pairs_' + str(min_pairs_per_cell_line) + '_' + str(max_pairs_per_cell_line) + '_th_'+str(threshold)+'_'+'neg_'+str(neg_fact)+'/'
    #
    #         if alg=='decagon':
    #             decagon_settings = config_map['ml_models_settings']['algs']['decagon']
    #             lr = decagon_settings['learning_rate']
    #             epochs =  decagon_settings['epochs']
    #             batch_size = decagon_settings['batch_size']
    #             dr = decagon_settings['dropout']
    #             use_drug_feat_options = decagon_settings['use_drug_feat']
    #             for drug_feat_option in use_drug_feat_options:
    #
    #
    #                 pos_out_file = out_dir + 'run_' + str(run_) + '/' + '/pos_val_scores' + '_drugfeat_' + str(drug_feat_option) +\
    #                                '_e_' + str(epochs) + '_lr_' + str(lr) + '_batch_' + str(batch_size) + '_dr_' + str(dr) + '.tsv'
    #                 neg_out_file = out_dir + 'run_' + str(run_) + '/' + '/neg_val_scores' + '_drugfeat_' + str(drug_feat_option) +\
    #                                '_e_' + str(epochs) + '_lr_' + str(lr) + '_batch_' + str(batch_size) + '_dr_' + str(dr) + '.tsv'
    #                 pos_df = pd.read_csv(pos_out_file , sep='\t')
    #                 neg_df = pd.read_csv(neg_out_file, sep='\t')
    #                 pos_neg_df = pd.concat([pos_df,neg_df], axis=0)\
    #                     [['drug_1','drug_2','cell_line', 'model_score','predicted','true']]
    #                 outputs_df[alg].append(pos_neg_df)
    #                 # pos_df_all_runs[alg].append(pos_df)
    #                 # neg_df_all_runs[alg].append(neg_df)
    #
    #                 print('plot: ')
    #                 title_suffix =  'run_' + str(run_) +  '_pairs_' + str(min_pairs_per_cell_line) + '_' + str(max_pairs_per_cell_line)+\
    #                                    '_drugfeat_' + str(drug_feat_option) +\
    #                                '_e_' + str(epochs) + '_lr_' + str(lr) + '_batch_' + str(batch_size) + '_dr_' + str(dr)
    #                 plot_dir = out_dir+'plot/'
    #                 metric_plot.plot_predicted_score_distribution(pos_df, neg_df, title_suffix,plot_dir)
    #                 metric_plot.plot_auprc_auroc(pos_neg_df, title_suffix, plot_dir)
        # metric_plot.performance_metric_evaluation_per_alg(outputs_df[alg], alg, config_map)





if __name__ == "__main__":
    config_map, kwargs = parse_args()
    main(config_map, **kwargs)

