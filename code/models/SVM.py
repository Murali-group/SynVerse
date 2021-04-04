import os, sys

import pandas as pd
import numpy as np
import pickle
import gzip
from scipy.sparse import csr_matrix

import matplotlib.pyplot as plt

# import utils
from models.utils import create_drug_drug_pairs_feature, create_syn_non_syn_feat_labels,save_drug_syn_probability

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))



def create_model(input_shape, layer_setup, lr, input_dropout, dropout, act_func):


    return model


def run_deepsynergy_model(drug_maccs_keys_targets_feature_df,\
                        synergy_df, non_synergy_df, cross_validation_folds,neg_cross_validation_folds, run_,out_dir, config_map):

    ## config parameters
    number_of_folds = config_map['ml_models_settings']['cross_val']['folds']
    ds_settings = config_map['ml_models_settings']['algs']['deepsynergy']
    layer_setups = ds_settings['layers']
    dropouts = ds_settings['dropout']
    input_dropouts = ds_settings['input_dropout']
    lrs = ds_settings['lr']
    act_func = ds_settings['act_func']
    epochs = ds_settings['epochs']
    batch_size = ds_settings['batch_size']

    cell_lines = synergy_df['Cell_line'].unique()
    cell_line_2_idx = {cell_line: i for i, cell_line in enumerate(cell_lines)}
    synergy_df['cell_line_idx'] = synergy_df['Cell_line'].apply(lambda x: cell_line_2_idx[x])
    non_synergy_df['cell_line_idx'] = non_synergy_df['Cell_line'].apply(lambda x: cell_line_2_idx[x])

    #one_hot_encoding of cell line feature
    cell_line_feat_df = pd.DataFrame(np.eye(len(cell_lines), dtype=int))


    for layer_setup in layer_setups:
        for lr in lrs:
            for input_dropout in input_dropouts:
                for dropout in dropouts:
                    predictions_dict = {'drug_1': [], 'drug_2': [], 'cell_line': [], 'predicted': [],
                                        'true': [], 'val_fold': []}
                    for fold in range(number_of_folds):

                        #########################  prepare feature and labels  ############################
                        training_synergy_df = synergy_df[~synergy_df.index.isin(cross_validation_folds[fold])] \
                            [['Drug1_pubchem_cid', 'Drug2_pubchem_cid', 'Cell_line', 'cell_line_idx']]

                        training_non_synergy_df = non_synergy_df[~non_synergy_df.index.isin(neg_cross_validation_folds[fold])] \
                            [['Drug1_pubchem_cid', 'Drug2_pubchem_cid', 'Cell_line', 'cell_line_idx']]

                        validation_synergy_df = synergy_df[synergy_df.index.isin(cross_validation_folds[fold])] \
                            [['Drug1_pubchem_cid', 'Drug2_pubchem_cid', 'Cell_line', 'cell_line_idx']]

                        validation_non_synergy_df = non_synergy_df[non_synergy_df.index.isin(neg_cross_validation_folds[fold])] \
                            [['Drug1_pubchem_cid', 'Drug2_pubchem_cid', 'Cell_line', 'cell_line_idx']]

                        train_feat_df, train_label = create_syn_non_syn_feat_labels(drug_maccs_keys_targets_feature_df, cell_line_feat_df, \
                                                                                 training_synergy_df, training_non_synergy_df)

                        val_feat_df, val_label = create_syn_non_syn_feat_labels(drug_maccs_keys_targets_feature_df, cell_line_feat_df, \
                                                                             validation_synergy_df, validation_non_synergy_df)

                        train_feat = csr_matrix(train_feat_df.values)
                        val_feat = csr_matrix(val_feat_df.values.astype(np.int32))
                        ##########################  train model  ##########################################

                        model = create_model(train_feat.shape[1], layer_setup, lr, input_dropout, dropout, act_func)

                        #train model
                        #save best model

                        predictions = best_model.predict(val_feat)
                        print('predictions: ', type(predictions), predictions.shape)

                        row=0
                        for drug_1, drug_2, cell_line in val_feat_df.index:
                            predictions_dict['drug_1'].append(drug_1)
                            predictions_dict['drug_2'].append(drug_2)
                            predictions_dict['cell_line'].append(cell_line)
                            predictions_dict['predicted'].append(predictions[row])
                            predictions_dict['true'].append(val_label[row])
                            predictions_dict['val_fold'].append(fold)

                            row+=1

                        train_feat = None
                        val_feat = None

                    predictions_df = pd.DataFrame.from_dict(predictions_dict)
                    pos_df = predictions_df[predictions_df['true'] == 1]
                    neg_df = predictions_df[predictions_df['true'] == 0]

                    save_drug_syn_probability(pos_df, neg_df, layer_setup, lr, input_dropout, dropout, batch_size, \
                                              epochs, act_func, run_, out_dir)
