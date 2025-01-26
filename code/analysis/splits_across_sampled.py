import pickle
import pandas as pd
import matplotlib.pyplot as plt
from plots.plot_utils import plot_double_dist, plot_dist
from scipy import stats
import numpy as np
from scipy.stats import skew, kurtosis
import os

def get_synergy_stat(df):
    drugs = set(df['drug_1_pid']).union(set(df['drug_2_pid']))
    cell_lines = set(df['cell_line_name'])
    triplets = set(zip(df['drug_1_pid'], df['drug_2_pid'], df['cell_line_name']))
    return triplets, drugs, cell_lines

def compare_data(df_1, df_2, prefix=''):
    '''compare df_1 and df_2.'''

    triplets1, drugs1, cell_lines1 = get_synergy_stat(df_1)
    triplets2, drugs2, cell_lines2 = get_synergy_stat(df_2)

    common_triplets = triplets1.intersection(triplets2)
    difference_in_triplets = triplets1.difference(triplets2)
    total_triplets = triplets1.union(triplets2)
    common_drugs = drugs1.intersection(drugs2)
    difference_in_drugs = drugs1.difference(drugs2)
    total_drugs = drugs1.union(drugs2)
    common_cell_lines = cell_lines1.intersection(cell_lines2)
    difference_in_cell_lines = cell_lines1.difference(cell_lines2)
    total_cell_lines = cell_lines1.union(cell_lines2)

    # print('common triplets: ', len(common_triplets), '   common drugs: ',len(common_drugs), '   common cell lines: ',len(common_cell_lines))
    # print('total triplets: ',len(total_triplets),'   total drugs: ', len(total_drugs),'   total cell lines: ', len(total_cell_lines))
    # print('different triplets: ', len(difference_in_triplets), '   different drugs: ', len(difference_in_drugs),'   different cell lines: ', len(difference_in_cell_lines))

    print(f"{prefix}\t{len(difference_in_triplets)}\t{len(difference_in_drugs)}\t{len(difference_in_cell_lines)}\t{len(total_triplets)}\t{len(total_drugs)}\t{len(total_cell_lines)}")


def compare_score_distributions(values_1, values_2, value_name_1, value_name_2):
    plot_double_dist(values_1, values_2, labels=[value_name_1, value_name_2])


def read_split_files(split_dir):
    all_data_file = split_dir + 'all.tsv'
    test_file = split_dir + 'test.tsv'
    train_val_file = split_dir + 'all_train.tsv'

    all_df = pd.read_csv(all_data_file, sep='\t')
    test_df = pd.read_csv(test_file, sep='\t')
    train_df = pd.read_csv(train_val_file, sep='\t')

    return all_df, test_df, train_df

def compute_skew_kurtosis(values):
    return skew(values), kurtosis(values)
def main():
    feature_sets = ['D_ECFP_4_MACCS_MFP_d1hot_mol_graph_smiles_C_c1hot', 'D_d1hot_C_c1hot_genex_genex_lincs_1000']
    split_types = ["leave_comb", "leave_drug", "leave_cell_line"]
    score_names = ["S_mean_mean", "synergy_loewe_mean"]
    print(f'split\trun_no\tdataset\tuncommon_triplets\tuncommon_drugs\tuncommon_cell\ttotal_triplets\ttotal_drugs\ttotal_cell\t')

    for feature_set in feature_sets:
        for split_type in split_types:
            for run_no in range(5):
                for score_name in score_names:
                    split_dir_orig = f"/home/grads/tasnina/Projects/SynVerse/inputs/splits/{feature_set}/k_0.05_{score_name}/{split_type}_0.2_0.25/run_{run_no}/"
                    split_dir_sampled = f"/home/grads/tasnina/Projects/SynVerse/inputs/splits/sample_norm_0.99/{feature_set}/k_0.05_{score_name}/{split_type}_0.2_0.25/run_{run_no}/"

                    if not os.path.exists(split_dir_sampled):
                        continue


                    all_df_orig, test_df_orig, train_df_orig = read_split_files(split_dir_orig)
                    all_df_sampled, test_df_sampled, train_df_sampled = read_split_files(split_dir_sampled)

                    prefix = f'{split_type}\t{run_no}\tALL  '
                    compare_data(all_df_orig,all_df_sampled, prefix)
                    prefix = f'{split_type}\t{run_no}\tTRAIN'
                    compare_data(train_df_orig, train_df_sampled, prefix)
                    prefix = f'{split_type}\t{run_no}\tTEST'
                    compare_data(test_df_orig, test_df_sampled, prefix)




main()