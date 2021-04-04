import os, sys

import pandas as pd
import numpy as np
import pickle
import gzip

def create_drug_drug_pairs_feature(drug_maccs_keys_targets_feature_df, cell_line_feat_df, synergy_df):
    drug_maccs_keys_targets_feature_df = drug_maccs_keys_targets_feature_df.set_index('pubchem_cid')

    drug_1_maccs_keys_targets = drug_maccs_keys_targets_feature_df.reindex(
        synergy_df['Drug1_pubchem_cid']).reset_index()
    drug_2_maccs_keys_targets = drug_maccs_keys_targets_feature_df.reindex(
        synergy_df['Drug2_pubchem_cid']).reset_index()
    cell_lines = cell_line_feat_df.reindex(synergy_df['cell_line_idx']).reset_index()

    drug_1_2_maccs_keys_targets_cell_lines = pd.concat(
        [drug_1_maccs_keys_targets, drug_2_maccs_keys_targets, cell_lines], axis=1). \
        set_index(['Drug1_pubchem_cid', 'Drug2_pubchem_cid', 'cell_line_idx'])
    drug_2_1_maccs_keys_targets_cell_lines = pd.concat(
        [drug_2_maccs_keys_targets, drug_1_maccs_keys_targets, cell_lines], axis=1). \
        set_index(['Drug1_pubchem_cid', 'Drug2_pubchem_cid', 'cell_line_idx'])

    # rename the columns before concatenation
    col_names_1 = list(drug_1_2_maccs_keys_targets_cell_lines.columns)
    col_names_2 = list(drug_2_1_maccs_keys_targets_cell_lines.columns)

    new_cols_1 = dict(zip(col_names_1, list(range(len(col_names_1)))))
    new_cols_2 = dict(zip(col_names_2, list(range(len(col_names_2)))))

    drug_1_2_maccs_keys_targets_cell_lines.rename(columns=new_cols_1, inplace=True)
    drug_2_1_maccs_keys_targets_cell_lines.rename(columns=new_cols_2, inplace=True)

    print(len(drug_1_2_maccs_keys_targets_cell_lines.columns), len(drug_2_1_maccs_keys_targets_cell_lines.columns))

    drugs_cell_lines = pd.concat([drug_1_2_maccs_keys_targets_cell_lines, drug_2_1_maccs_keys_targets_cell_lines], \
                                 axis=0)

    return drugs_cell_lines


def create_syn_non_syn_feat_labels(drug_maccs_keys_targets_feature_df, cell_line_feat_df, \
                                   synergy_df, non_synergy_df):
    # retun 1 df and 1 numpy arrays:
    # 1. feat_df = contains feature for both postive and negative drug pairs in both drugA-drugB and drugB-drugA order
    # 2. labels: contains 0/1 label for each drug-drug-cell_line triplets.

    feat_synergy_pairs_df = create_drug_drug_pairs_feature(drug_maccs_keys_targets_feature_df, cell_line_feat_df, \
                                                           synergy_df)
    label_1 = np.ones(len(feat_synergy_pairs_df))
    feat_non_synergy_pairs_df = create_drug_drug_pairs_feature(drug_maccs_keys_targets_feature_df, cell_line_feat_df, \
                                                               non_synergy_df)
    label_0 = np.zeros(len(feat_non_synergy_pairs_df))

    feat_df = pd.concat([feat_synergy_pairs_df, feat_non_synergy_pairs_df], axis=0)

    # convert to numpy array
    # feat = feat_df.values
    label = np.concatenate((label_1, label_0), axis=0)
    return feat_df, label


def save_drug_syn_probability(pos_df, neg_df, layer_setup, lr, input_dropout, dropout, batch_size, epochs, act_func,
                              run_, out_dir):
    # inputs: df with link prediction probability after applying sigmoid on models predicted score for an edges(positive, negative)
    pos_out_file = out_dir + 'run_' + str(run_) + '/' + \
                   '/pos_val_scores' + '_layers_' + str(layer_setup) + '_e_' + str(epochs) + '_lr_' + str(
        lr) + '_batch_' + str(batch_size) + '_dr_' + \
                   str(input_dropout) + '_' + str(dropout) + '_act_' + str(act_func) + '.tsv'

    neg_out_file = out_dir + 'run_' + str(run_) + '/' + \
                   '/neg_val_scores' + '_layers_' + str(layer_setup) + '_e_' + str(epochs) + '_lr_' + str(
        lr) + '_batch_' + str(batch_size) + '_dr_' + \
                   str(input_dropout) + '_' + str(dropout) + '_act_' + str(act_func) + '.tsv'

    os.makedirs(os.path.dirname(pos_out_file), exist_ok=True)
    os.makedirs(os.path.dirname(neg_out_file), exist_ok=True)

    pos_df.to_csv(pos_out_file, sep='\t')
    neg_df.to_csv(neg_out_file, sep='\t')