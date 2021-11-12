import os, sys

import pandas as pd
import numpy as np
import pickle
import gzip
from scipy.sparse import csr_matrix

import matplotlib.pyplot as plt
import time

# import utils
from models.utils import create_drug_drug_pairs_feature, create_syn_non_syn_feat_labels
from sklearn.svm import SVC
import  sklearn.metrics as metrics



def save_test_drug_syn_probability(pos_df, neg_df, model_info, use_genex, use_target, out_dir):


    pos_out_file = out_dir + \
                   '/pos_val_scores_' +model_info +'_use_genex_' + str(use_genex)+\
                   '_use_target_' + str(use_target)+'.tsv'

    neg_out_file = out_dir + \
                   '/neg_val_scores_' + model_info+'_use_genex_' + str(use_genex) +\
                   '_use_target_' + str(use_target)+'.tsv'

    os.makedirs(os.path.dirname(pos_out_file), exist_ok=True)
    os.makedirs(os.path.dirname(neg_out_file), exist_ok=True)

    pos_df.to_csv(pos_out_file, sep='\t')
    neg_df.to_csv(neg_out_file, sep='\t')

def save_val_eval(val_f1_macro,val_f1_micro, val_precision, val_recall, model_info, use_genex, use_target,out_dir, fold_no):
    model_info = model_info + '_use_genex_' + str(use_genex) + '_use_target_' + str(use_target)+'_'
    val_file = out_dir +model_info+ 'model_val_loss .txt'

    os.makedirs(os.path.dirname(val_file), exist_ok=True)

    if fold_no==0:
        val_file  = open(val_file, 'w')
    else:
        val_file  = open(val_file, 'a')


    # val_file.write(model_info)
    val_file.write('\nval_f1_macro: '+ str(val_f1_macro))
    val_file.write('\nval_f1_micro: ' + str(val_f1_micro))
    val_file.write('\nval_precision_macro: ' + str(val_precision))
    val_file.write('\nval_recall_macro: ' + str(val_recall))
    val_file.write('\n\n')

    val_file.close()

def run_svm_model(drug_maccs_keys_targets_feature_df,drug_maccs_keys_feature_df,\
                            gene_expression_feature_df,use_genex,use_target,
                              synergy_df, non_synergy_df,cell_line_2_idx,idx_2_cell_line,
                              cross_validation_folds,
                              neg_cross_validation_folds,
                              svm_params, out_dir, config_map):
    '''
    cross_validation_folds is in the form:
    cross_validation_folds = {i: {'test': [], 'train': [], 'val': []} for i in range(number_of_folds)}
    '''


    ## config parameters
    number_of_folds = config_map['ml_models_settings']['cross_val']['folds']
    cell_lines = synergy_df['Cell_line'].unique()
    synergy_df['cell_line_idx'] = synergy_df['Cell_line'].apply(lambda x: cell_line_2_idx[x])
    non_synergy_df['cell_line_idx'] = non_synergy_df['Cell_line'].apply(lambda x: cell_line_2_idx[x])

    if use_genex:
        # gene expression feat for cell lines
        cell_line_feat_df = gene_expression_feature_df

    else:
        #one_hot_encoding of cell line feature
        cell_line_feat_df = pd.DataFrame(np.eye(len(cell_lines), dtype=int))


    count=0
    c = svm_params['c']
    kernel = svm_params['kernel']
    gamma = svm_params['gamma']


    #DATA Preparation
    test_predictions_dict = {'drug_1': [], 'drug_2': [], 'cell_line': [], 'predicted': [],
                        'true': []}
    val_predictions_dict = {'drug_1': [], 'drug_2': [], 'cell_line': [], 'predicted': [],
                             'true': []}


    for fold in range(number_of_folds):
        t1= time.time()

        #########################  prepare feature and labels  ############################
        training_synergy_df = synergy_df[synergy_df.index.isin(cross_validation_folds[fold]['train'])] \
            [['Drug1_pubchem_cid', 'Drug2_pubchem_cid', 'Cell_line', 'cell_line_idx']]

        training_non_synergy_df = non_synergy_df[non_synergy_df.index.isin(neg_cross_validation_folds[fold]['train'])] \
            [['Drug1_pubchem_cid', 'Drug2_pubchem_cid', 'Cell_line', 'cell_line_idx']]

        validation_synergy_df = synergy_df[synergy_df.index.isin(cross_validation_folds[fold]['val'])] \
            [['Drug1_pubchem_cid', 'Drug2_pubchem_cid', 'Cell_line', 'cell_line_idx']]

        validation_non_synergy_df = non_synergy_df[non_synergy_df.index.isin(neg_cross_validation_folds[fold]['val'])] \
            [['Drug1_pubchem_cid', 'Drug2_pubchem_cid', 'Cell_line', 'cell_line_idx']]

        validation_es_synergy_df = synergy_df[synergy_df.index.isin(cross_validation_folds[fold]['val_es'])] \
            [['Drug1_pubchem_cid', 'Drug2_pubchem_cid', 'Cell_line', 'cell_line_idx']]

        validation_es_non_synergy_df = non_synergy_df[non_synergy_df.index.isin(neg_cross_validation_folds[fold]['val_es'])] \
            [['Drug1_pubchem_cid', 'Drug2_pubchem_cid', 'Cell_line', 'cell_line_idx']]

        test_synergy_df = synergy_df[synergy_df.index.isin(cross_validation_folds[fold]['test'])] \
            [['Drug1_pubchem_cid', 'Drug2_pubchem_cid', 'Cell_line', 'cell_line_idx']]

        test_non_synergy_df = non_synergy_df[non_synergy_df.index.isin(neg_cross_validation_folds[fold]['test'])] \
            [['Drug1_pubchem_cid', 'Drug2_pubchem_cid', 'Cell_line', 'cell_line_idx']]


        if use_target:

            train_feat_df, train_label = create_syn_non_syn_feat_labels(drug_maccs_keys_targets_feature_df, cell_line_feat_df, \
                                                                     training_synergy_df, training_non_synergy_df)

            val_feat_df, val_label = create_syn_non_syn_feat_labels(drug_maccs_keys_targets_feature_df, cell_line_feat_df, \
                                                                 validation_synergy_df, validation_non_synergy_df)

            val_es_feat_df, val_es_label = create_syn_non_syn_feat_labels(drug_maccs_keys_targets_feature_df,
                                                                    cell_line_feat_df, \
                                                                    validation_es_synergy_df, validation_es_non_synergy_df)
            test_feat_df, test_label = create_syn_non_syn_feat_labels(drug_maccs_keys_targets_feature_df,
                                                                      cell_line_feat_df,
                                                                      test_synergy_df, test_non_synergy_df)
        else:
            train_feat_df, train_label = create_syn_non_syn_feat_labels(drug_maccs_keys_feature_df,
                                                                        cell_line_feat_df, \
                                                                        training_synergy_df, training_non_synergy_df)

            val_feat_df, val_label = create_syn_non_syn_feat_labels(drug_maccs_keys_feature_df,
                                                                    cell_line_feat_df, \
                                                                    validation_synergy_df, validation_non_synergy_df)
            val_es_feat_df, val_es_label = create_syn_non_syn_feat_labels(drug_maccs_keys_feature_df,
                                                                    cell_line_feat_df, \
                                                                    validation_es_synergy_df, validation_es_non_synergy_df)
            test_feat_df, test_label = create_syn_non_syn_feat_labels(drug_maccs_keys_feature_df,
                                                                      cell_line_feat_df,
                                                                      test_synergy_df, test_non_synergy_df)

        train_feat = csr_matrix(train_feat_df.values)
        val_feat = csr_matrix(val_feat_df.values)
        # val_es_feat = csr_matrix(val_es_feat_df.values)
        test_feat = csr_matrix(test_feat_df.values)


        ##########################  train model  ##########################################

        svc = SVC(C=c, kernel=kernel, gamma=gamma)
        model = svc.fit(train_feat, train_label)

        #prediction on val
        val_predictions= model.predict(val_feat)

        val_f1_macro = metrics.f1_score(val_label, val_predictions, average='macro')

        val_f1_micro = metrics.f1_score(val_label, val_predictions, average='micro')

        val_precision_macro = metrics.precision_score(val_label, val_predictions, average='macro')

        val_recall_macro = metrics.recall_score(val_label, val_predictions, average='macro')

        #prediction on test
        test_predictions = model.predict(test_feat)
        row=0
        for drug_1, drug_2, cell_line_idx in test_feat_df.index:
            test_predictions_dict['drug_1'].append(drug_1)
            test_predictions_dict['drug_2'].append(drug_2)
            test_predictions_dict['cell_line'].append(idx_2_cell_line[cell_line_idx])
            test_predictions_dict['predicted'].append(test_predictions[row])
            test_predictions_dict['true'].append(test_label[row])
            # predictions_dict['fold'].append(fold)
            row+=1

        # free memory
        del train_feat
        del val_feat
        del test_feat

        print('done: ', fold, 'duration: ', time.time()-t1)
    test_predictions_df = pd.DataFrame.from_dict(test_predictions_dict)
    test_pos_df = test_predictions_df[test_predictions_df['true'] == 1]
    test_neg_df = test_predictions_df[test_predictions_df['true'] == 0]

    model_info = 'c_' + str(c) + '_kernel_' + kernel + '_gamma_' + gamma
    save_test_drug_syn_probability(test_pos_df, test_neg_df, model_info, use_genex, use_target, out_dir)
    save_val_eval(val_f1_macro, val_f1_micro, val_precision_macro, val_recall_macro, model_info, use_genex, use_target, out_dir, fold)

    count+=1