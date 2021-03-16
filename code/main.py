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
import os
import sys
#from tqdm import tqdm
import copy
import time
import numpy as np
#from scipy import sparse
from scipy.io import savemat, loadmat
import pandas as pd
#import subprocess
# sys.path.insert(0, '/home/tasnina/Projects/Synverse/')
import cross_validation_minibatch as cross_val
import models.decagon_handler.run_decagon as decagon
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
    group.add_argument('--config', type=str, default="/home/tasnina/Projects/SynVerse/config-files/master-config.yaml",
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

def preprocess_inputs(config_map, string_cutoff):
    
    project_dir = config_map['project_dir']
    synergy_file = project_dir + config_map['inputs']['synergy']
    drug_target_file = project_dir + config_map['inputs']['drug']['target']
    
    # mat: /c"+str(string_cutoff)+"_combined_score_sparse_net.mat"
        # nodes: "inputs/networks/c"+str(string_cutoff)+"_combined_score_node_ids.txt"
    ppi_network_file = project_dir + config_map['inputs']['ppi']+ "c" + str(string_cutoff)+"_combined_score_sparse_net.mat"
    
    # ppi_network_file = '/home/tasnina/Projects/SynVerse/inputs/networks/c700_combined_score_sparse_net.mat'
    
    ppi_node_to_idx_file = project_dir +config_map['inputs']['ppi']+ "c" + str(string_cutoff)+"_combined_score_node_ids.txt"
    

    minimum_number_of_synergistic_pairs_per_cell_line = config_map['synergy_data_settings']['min_pairs']


    ppi_sparse_matrix = loadmat(ppi_network_file)['Networks'][0][0]

    print(ppi_sparse_matrix.shape)

    gene_node_2_idx = pd.read_csv(ppi_node_to_idx_file, sep='\t', header=None, names=['gene','index'])
    gene_node_2_idx= dict(zip(gene_node_2_idx['gene'], gene_node_2_idx['index']))
    
    
    
    
    ##***load drug-drug synergy dataset. This contains drugs for which we have atleast one target info before removing non-PPI ##**targtets.
    #load drug-drug synergy dataset. This contains drugs for which we have atleast one target info before removing non-PPI targtets.
    synergy_df = pd.read_csv(synergy_file, sep = '\t',dtype={'Drug1_pubchem_cid':str,\
                'Drug2_pubchem_cid': str, 'Cell_line': str, 'Loewe': np.float64,\
                'Bliss': np.float64, 'ZIP': np.float64})

    # print(synergy_df.head())

    
    #threshold on Loewe score to determine if two drugs are synergistic or not

    if(config_map['synergy_data_settings']['threshold']['should_apply']==True):
        threshold = config_map['synergy_data_settings']['threshold']['val']
        synergy_df['Loewe_label'] = synergy_df['Loewe'].astype(np.float64).apply(lambda x: 0 if x<threshold else 1)
        synergy_df = synergy_df[synergy_df['Loewe_label']==1]



    cell_lines  = synergy_df['Cell_line'].unique()
    drug_pairs_per_cell_line = {x : 0 for x in cell_lines}
    
    for row in synergy_df.itertuples():
            drug_pairs_per_cell_line[row.Cell_line] += 1
    
    print('no of cell lines before: ', len(synergy_df['Cell_line'].unique()))
    
    for cell_line in cell_lines:
        if drug_pairs_per_cell_line[cell_line]<minimum_number_of_synergistic_pairs_per_cell_line:
            synergy_df = synergy_df[synergy_df['Cell_line']!=cell_line]
    
    # print('no of cell lines after: ', len(synergy_df['Cell_line'].unique()))
    
    synergistics_drugs = set(list(synergy_df['Drug1_pubchem_cid'])).union(set(list(synergy_df['Drug2_pubchem_cid'])))
    # print('number of drugs after applying threshold on synergy data:' , len(synergistics_drugs))
    
    
    
    ##***create drug-target network. This 'drug_target_map_filtered.tsv' contains target info for drugs which are in synergy dataset
    drug_target_df_init = pd.read_csv(drug_target_file, sep ='\t', index_col = 0, header = 0, dtype = str)
    
    #mapping genes to their index as in ppi
    genes_in_ppi = gene_node_2_idx.keys()
    genes_as_drug_target = list(drug_target_df_init['uniprot_id'])
    
    targets_not_in_ppi = list(set(genes_as_drug_target)- set(genes_in_ppi))
    # print('Drug targets not present in PPI:', len(targets_not_in_ppi), targets_not_in_ppi)
    
    print('\nCheck if any drug has been removed after excluding non PPI proteins:\n')
    # print('Before: ', drug_target_df_init.nunique())
    #for now remove the 105 targets which do not have ppi data, from drug target list
    #this will also remove any drug that has no target remianing after removing the non PPI targets.
    drug_target_df = drug_target_df_init[~drug_target_df_init['uniprot_id'].isin(targets_not_in_ppi)]
    # print('removed_drug for not being in ppi:'set(drug_target_df_init['pubchem_cid']).difference(set(drug_target_df['pubchem_cid'])))
    
    
    #find common drugs between thresholded synergy_df and filtered drug_target_df
    drugs_in_drugcombdb = set(synergy_df['Drug1_pubchem_cid'].astype(str)).union\
                                (set(synergy_df['Drug2_pubchem_cid'].astype(str)))
    drugs_having_target_info = set(drug_target_df['pubchem_cid'].astype(str))
    common = list(drugs_in_drugcombdb.intersection(drugs_having_target_info))
    print('total drugcombdb drug: ', len(drugs_in_drugcombdb))
    print('target info present for: ', len(drugs_having_target_info))
    print('common: ',len(common))
    
    
    #keep only the common drugs in synergy_df and drug_target_df
    drug_target_df = drug_target_df[drug_target_df['pubchem_cid'].isin(common)]
    synergy_df = synergy_df[synergy_df['Drug1_pubchem_cid'].isin(common) & synergy_df['Drug2_pubchem_cid'].isin(common)]

    # add index column to synergy_df
    synergy_df.set_index(pd.Series(range(len(synergy_df))), inplace=True)

    return ppi_sparse_matrix, gene_node_2_idx, drug_target_df, synergy_df
    
        
    
def main(config_map, **kwargs):
    string_cutoff = config_map['string_network_preparation_settings']['string_cutoff']
    ppi_sparse_matrix, gene_node_2_idx, drug_target_df, synergy_df = preprocess_inputs(config_map,string_cutoff)
    
    cross_val_types = config_map['ml_models_settings']['cross_val']['types']
    number_of_folds = config_map['ml_models_settings']['cross_val']['folds']

    #dict of dict to hold different types of cross val split folds
    type_wise_cross_val_folds = {cross_val_type: dict() for cross_val_type in cross_val_types }
    for cross_val_type in cross_val_types:
        type_wise_cross_val_folds[cross_val_type] = cross_val.\
            create_cross_val_folds(synergy_df, cross_val_type, number_of_folds)
    algs = config_map['ml_models_settings']['algs']
    for alg in algs:
        if alg=='decagon':
            if algs[alg]['should_run']==True:
                print('Model running: ', alg)
                decagon.run_decagon_model(ppi_sparse_matrix,gene_node_2_idx,drug_target_df,synergy_df,\
                                  type_wise_cross_val_folds['stratified'], config_map)



if __name__ == "__main__":
    config_map, kwargs = parse_args()
    main(config_map, **kwargs)

