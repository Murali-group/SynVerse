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
import matplotlib.pyplot as plt
#import subprocess
# sys.path.insert(0, '/home/tasnina/Projects/Synverse/')
import cross_validation as cross_val
import models.synverse.run_synverse as synverse
import evaluation.evaluation_handler as evaluation_handler
import utils
# import models.deepsynergy as deepsynergy
#
# import models.decagon_handler.run_decagon as decagon


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

    group.add_argument('--exp-score', type=str, default='Z_SCORE',
                       help="gene expression score to consider. Options: 'Z_SCORE', 'REGULATION' ")

    #cross validation arguments
    group.add_argument('--cvdir', type=str, default="genex_4",
                       help="folder to save cross validation folds ")
    #neagtive sampling arguments
    group.add_argument('--sampling', type=str, default="semi_random",
                       help="two types of negative sampling: 'semi_random' and 'degree_based' ")



    # synverse model parameters
    group.add_argument('--dddecoder', type=str,
                       default='nndecoder',
                       help="decoder for drug drug edges")
    group.add_argument('--encoder', type=str,
                       default='local',
                       help="encoder: 'local' is separate weight matrix for each cell line or 'global' if one\
                            weight matrix is used for all cell lines")

    group.add_argument('--drug_based_batch_end', action = 'store_true',
                       help="if true, at each epoch once all the drug_drug batches are used for training, the epoch ends")

    #evaluation arguments
    group.add_argument('--recall', type=float,
                       default=0.3,
                       help="recall value for early precision")

    group.add_argument('--force-cvdir', action = 'store_true')
    group.add_argument('--train', action = 'store_false')
    group.add_argument('--eval', action = 'store_false')

    return parser


def prepare_synergy_pairs(synergy_df,number_of_top_cell_lines,top_k_percent ):
    ##***load drug-drug synergy dataset. This contains drugs for which we have atleast one target info before removing non-PPI targets
    # synergy_df = pd.read_csv(synergy_file, sep='\t', dtype={'Drug1_pubchem_cid': str,
    #                                                         'Drug2_pubchem_cid': str,
    #                                                         'Cell_line': str,
    #                                                         'Loewe': np.float64,
    #                                                         'Bliss': np.float64,
    #                                                         'ZIP': np.float64})

    ################based on top k percent###################
    cell_lines = synergy_df['Cell_line'].unique()
    drug_pairs_per_cell_line = {x: 0 for x in cell_lines}
    for row in synergy_df.itertuples():
        drug_pairs_per_cell_line[row.Cell_line] += 1
    drug_pairs_per_cell_line = dict(sorted(drug_pairs_per_cell_line.items(),\
                                           key=lambda item: item[1], reverse=True))


    top_k_cell_lines = list(drug_pairs_per_cell_line.keys())[0:number_of_top_cell_lines]



    synergy_df_new = pd.DataFrame()
    cell_line_wise_threshold = {x:0 for x in top_k_cell_lines}
    cell_line_wise_drug_pairs = {x:0 for x in top_k_cell_lines}

    #keep only the top k cell lines
    for cell_line in top_k_cell_lines:
        cell_line_df = synergy_df[synergy_df['Cell_line']==cell_line]
        #
        # hist_data = list(cell_line_df['Loewe'])
        # plt.hist(hist_data, weights=np.zeros_like(hist_data) + 1. / len(hist_data), bins=10)
        # plt.title(cell_line)
        # plt.show()
        # plt.clf()

        cell_line_df.sort_values(by='Loewe', inplace= True, ascending=False)
        cell_line_df.reset_index(drop = True, inplace= True)

        number_of_pairs_in_top_k_percent = (top_k_percent/100.0) * len(cell_line_df)
        cell_line_df = cell_line_df.loc[0:number_of_pairs_in_top_k_percent]

        cell_line_wise_threshold[cell_line] = cell_line_df['Loewe'].min()
        cell_line_wise_drug_pairs[cell_line] = number_of_pairs_in_top_k_percent

        # hist_data = list(cell_line_df['Loewe'])
        # plt.hist(hist_data, weights=np.zeros_like(hist_data) + 1. / len(hist_data), bins=10)
        # plt.title(cell_line)
        # plt.show()
        # plt.clf()

        synergy_df_new = pd.concat([synergy_df_new, cell_line_df], axis = 0, ignore_index=True)
    synergy_df_new['Loewe_label'] =  pd.Series(np.ones(len(synergy_df_new)), dtype=int)
    # print(synergy_df_new.head())

    plt.scatter(cell_line_wise_threshold.keys(), cell_line_wise_threshold.values())
    plt.xticks(rotation='vertical', fontsize=9)
    plt.title('Cell line wise thresholds for being considered as synergistic')
    plt.tight_layout(pad=1, w_pad=0.4, h_pad=0.5)
    plt.show()
    plt.clf()

    plt.scatter(cell_line_wise_drug_pairs.keys(), cell_line_wise_drug_pairs.values(), vmin=0)
    plt.xticks(rotation='vertical', fontsize=9)
    plt.title('Cell line wise synergistic drug pairs')
    plt.tight_layout(pad=1, w_pad=0.4, h_pad=0.5)
    plt.show()
    plt.clf()

    return synergy_df_new


def preprocess_inputs(string_cutoff,minimum_number_of_synergistic_pairs_per_cell_line,maximum_number_of_synergistic_pairs_per_cell_line,
                      synergy_df, config_map):
    #outputs: ppi_sparse_matrix: protein-protein interaction matrix from STRING with cutoff at 700
    # gene_node_2_idx (dictionray): PPI gene to index
    #the following three datastructures we do not have such a drug which is present in one but not in other two. The drugs are such that\
    #for the drug has at least one target in PPI, is present in synergy_df, has maccs_keys feature available.
    # drug_target_df:
    # drug_maccs_keys_feature_df:
    # synergy_df:
    project_dir = config_map['project_dir']
    synergy_file = project_dir + config_map['inputs']['synergy']
    number_of_top_cell_lines = config_map['synergy_data_settings']['number_of_top_cell_lines']
    top_k_percent = config_map['synergy_data_settings']['top_k_percent_pairs']

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
    
        
def preprocess_drug_feat(synergy_df, config_map):
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
    drug_maccs_keys_targets_feature_df = pd.read_csv(drug_maccs_keys_targets_file, sep='\t', index_col=None,\
                                                     dtype={'pubchem_cid': str})
    drug_maccs_keys_targets_feature_df = drug_maccs_keys_targets_feature_df[drug_maccs_keys_targets_feature_df['pubchem_cid'].\
        isin(drugs_in_synergy_df)]

    return drug_maccs_keys_feature_df, drug_maccs_keys_targets_feature_df



def main(config_map, **kwargs):

    print(kwargs.get('force_cvdir'), kwargs.get('train'), kwargs.get('eval'))
    force_cvdir = kwargs.get('force_cvdir')
    # print(force_run)


    #synergy data settings
    string_cutoff = config_map['string_network_preparation_settings']['string_cutoff']
    min_pairs_per_cell_line = config_map['synergy_data_settings']['min_pairs']
    max_pairs_per_cell_line =  config_map['synergy_data_settings']['max_pairs']
    threshold = config_map['synergy_data_settings']['threshold']['val']
    number_of_top_cell_lines = config_map['synergy_data_settings']['number_of_top_cell_lines']
    top_k_percent = config_map['synergy_data_settings']['top_k_percent_pairs']

    #cell line specific gene expression data settings
    exp_score = kwargs.get('exp_score')
    init_gene_expression_file_path =  config_map['project_dir']+config_map['inputs']['cell_lines'][exp_score]
    compressed_gene_expression_dir =  os.path.dirname(init_gene_expression_file_path)
    num_compressed_gene = config_map['gene_expression_data_settings']['num_compressed_gene']

    # cross val settings
    test_frac = config_map['ml_models_settings']['cross_val']['test_frac']
    cross_val_types = config_map['ml_models_settings']['cross_val']['types']
    number_of_folds = config_map['ml_models_settings']['cross_val']['folds']
    neg_fact = config_map['ml_models_settings']['cross_val']['neg_fact']
    neg_sampling_type = kwargs.get('sampling')
    number_of_runs = config_map['ml_models_settings']['cross_val']['runs']
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
    gene_expression_file = config_map['project_dir'] + config_map['inputs']['cell_lines']['compressed_gene_expression']
    gene_expression_feature_df = pd.read_csv(gene_expression_file, sep='\t', index_col=0)

    #take a few cell lines for the time being. so running the code is faster
    # n_gene_ex_cell_lines_to_consider = 10
    # filtered_cell_lines_in_gene_expresion_df = list(gene_expression_feature_df['cell_line_name'].unique())[0:n_gene_ex_cell_lines_to_consider]
    # gene_expression_feature_df = gene_expression_feature_df[gene_expression_feature_df['cell_line_name'].\
    #                             isin(filtered_cell_lines_in_gene_expresion_df)]

    # only keep the cell_lines for which we have gene expression value available
    synergy_df = synergy_df[synergy_df['Cell_line'].isin(list(gene_expression_feature_df['cell_line_name']))]
    synergy_df = prepare_synergy_pairs(synergy_df, number_of_top_cell_lines, top_k_percent)
    gene_expression_feature_df=gene_expression_feature_df[gene_expression_feature_df['cell_line_name'].\
                                isin(list(synergy_df['Cell_line']))]

    #now map cell line names to index for future use in different models
    cell_line_2_idx, idx_2_cell_line = utils.generate_cell_line_idx_mapping(synergy_df)
    gene_expression_feature_df['cell_line_index'] = gene_expression_feature_df['cell_line_name'].\
                                                    apply(lambda x: cell_line_2_idx[x])
    gene_expression_feature_df.drop('cell_line_name', axis=1, inplace=True)
    gene_expression_feature_df.set_index('cell_line_index', inplace=True, drop=True)


    #procesing network data
    ppi_sparse_matrix, gene_node_2_idx, drug_target_df, synergy_df = \
        preprocess_inputs(string_cutoff,min_pairs_per_cell_line,max_pairs_per_cell_line,synergy_df, config_map)



    #processing durg feature data
    drug_maccs_keys_feature_df,drug_maccs_keys_targets_feature_df = preprocess_drug_feat(synergy_df, config_map)



    
    should_run_algs = []
    for alg in algs:
        if algs[alg]['should_run'] == True:
            should_run_algs.append(alg)
    print('should_run_algs: ', should_run_algs)

    # dict of dict to hold different types of cross val split folds
    type_wise_pos_cross_val_folds = {cross_val_type: dict() for cross_val_type in cross_val_types}
    type_wise_neg_cross_val_folds = {cross_val_type: dict() for cross_val_type in cross_val_types}

    # generate model prediction and result
    if kwargs.get('train')==True:
        for run_ in range(number_of_runs):
            print("RUN NO:", run_)
            #train-test split
            # train_synergy_df, test_synergy_df = cross_val.train_test_split(synergy_df, test_frac)

            for cross_val_type in cross_val_types:
                cross_val_dir = config_map['project_dir'] + config_map['output_dir'] + 'cross_val/' + cross_val_type + '/' + \
                                'pairs_' + str(min_pairs_per_cell_line) + '_' + \
                                str(max_pairs_per_cell_line) + '_th_' + str(threshold)+'_cell_lines_' + str(number_of_top_cell_lines)+ \
                                '_percent_'+ str(top_k_percent)+\
                                '_' + 'neg_' + str(neg_fact) + '_'+ neg_sampling_type +'_neg_sampling_' + kwargs.get('cvdir') + '/run_' + str(run_) + '/'

                pos_fold_file = cross_val_dir + 'pos_folds.tsv'
                neg_fold_file = cross_val_dir + 'neg_folds.tsv'
                # syn_df_file = cross_val_dir + 'synergy.tsv'
                non_syn_file = cross_val_dir + 'non_synergy.tsv'

                if (not os.path.exists(pos_fold_file))|(not os.path.exists(neg_fold_file))|(not os.path.exists(non_syn_file))|(force_cvdir==True):
                    type_wise_pos_cross_val_folds[cross_val_type], type_wise_neg_cross_val_folds[cross_val_type],\
                    non_synergy_df = cross_val.\
                    create_cross_val_folds(synergy_df, cross_val_type, number_of_folds, neg_fact, neg_sampling_type)

                    #save the output from cross-val
                    os.makedirs(os.path.dirname(pos_fold_file), exist_ok=True)
                    pd.DataFrame.from_dict(type_wise_pos_cross_val_folds[cross_val_type], orient = 'index').\
                        to_csv(pos_fold_file, sep='\t', index=True)
                    pd.DataFrame.from_dict(type_wise_neg_cross_val_folds[cross_val_type],orient='index').\
                        to_csv(neg_fold_file, sep='\t', index=True)
                    # synergy_df.to_csv(syn_df_file, sep='\t')
                    non_synergy_df.to_csv(non_syn_file, index = True, sep='\t')

                pos_fold_df = pd.read_csv(pos_fold_file, sep='\t', index_col=0)
                neg_fold_df = pd.read_csv(neg_fold_file, sep='\t',index_col=0)
                non_synergy_df = pd.read_csv(non_syn_file,  sep = '\t', index_col = 0, dtype={'Drug1_pubchem_cid': str,\
                                                    'Drug2_pubchem_cid':str, 'Cell_line': str, 'Loewe_label':int})

                for fold in range(number_of_folds):
                    type_wise_pos_cross_val_folds[cross_val_type][fold] = list(pos_fold_df.loc[fold].dropna().astype(int))
                    type_wise_neg_cross_val_folds[cross_val_type][fold] = list(neg_fold_df.loc[fold].dropna().astype(int))
                # for i in range(number_of_folds):
                #     print('cross val', len(type_wise_pos_cross_val_folds[cross_val_type][i]))

                print('final number of drug pairs going into training: ', len(synergy_df))
                for alg in should_run_algs:


                    out_dir = config_map['project_dir'] + config_map['output_dir'] + alg + '/' + cross_val_type + '/' + \
                                    'pairs_' + str(min_pairs_per_cell_line) + '_' + \
                                    str(max_pairs_per_cell_line) + '_th_' + str(threshold) + '_cell_lines_' + str(number_of_top_cell_lines) + \
                                    '_percent_' + str(top_k_percent) + \
                                    '_' + 'neg_' + str(neg_fact) + '_' + neg_sampling_type + '_neg_sampling_' + kwargs.get('cvdir')+'/'+'run_' + str(run_)+'/'

                    os.makedirs(out_dir, exist_ok=True)


                    if alg=='synverse':
                            print('Model running: ', alg)
                            encoder_type = kwargs.get('encoder')
                            dd_decoder_type = kwargs.get('dddecoder')

                            synverse.run_synverse_model(ppi_sparse_matrix, gene_node_2_idx, drug_target_df,
                                        drug_maccs_keys_feature_df, gene_expression_feature_df, synergy_df, non_synergy_df,
                                        cell_line_2_idx, idx_2_cell_line,
                                        type_wise_pos_cross_val_folds[cross_val_type], type_wise_neg_cross_val_folds[cross_val_type],
                                        cross_val_dir, neg_sampling_type, encoder_type, dd_decoder_type, out_dir, config_map,
                                        use_drug_based_batch_end = kwargs.get('drug_based_batch_end'))


                    if alg=='deepsynergy':
                            print('Model running: ', alg)
                            # deepsynergy.run_deepsynergy_model(drug_maccs_keys_targets_feature_df,\
                            #     train_synergy_df, test_synergy_df, non_synergy_df, type_wise_pos_cross_val_folds['random'],type_wise_neg_cross_val_folds['random'], i,out_dir, config_map)
                            deepsynergy.run_deepsynergy_model(drug_maccs_keys_targets_feature_df, \
                                                              synergy_df, non_synergy_df,
                                                              type_wise_pos_cross_val_folds[cross_val_type],
                                                              type_wise_neg_cross_val_folds[cross_val_type], out_dir, config_map)

                    if alg == 'decagon':
                        print('Model running: ', alg)
                        decagon.run_decagon_model(ppi_sparse_matrix, gene_node_2_idx, \
                                                  drug_target_df, drug_maccs_keys_feature_df, \
                                                  synergy_df, non_synergy_df,
                                                  type_wise_pos_cross_val_folds[cross_val_type],
                                                  type_wise_neg_cross_val_folds[cross_val_type], \
                                                  out_dir, config_map)

    ################### PLOT ######################################
    if kwargs.get('eval') == True:
        for cross_val_type in cross_val_types:
            if kwargs.get('drug_based_batch_end'):
                extra_direction_on_out_dir = 'drug_based_batch_end/'
            else:
                extra_direction_on_out_dir = ''
            evaluation_handler.evaluate(should_run_algs,cross_val_type, kwargs, config_map, extra_direction_on_out_dir)

if __name__ == "__main__":
    config_map, kwargs = parse_args()
    main(config_map, **kwargs)

