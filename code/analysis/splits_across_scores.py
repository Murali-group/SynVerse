import pickle
import pandas as pd
import matplotlib.pyplot as plt
from plots.plot_utils import plot_double_dist, plot_dist
from scipy import stats
import numpy as np
from scipy.stats import skew, kurtosis


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



def filter_normal_conforming_data(values, retain_ratio=0.8, plot=False):
    """
    Filters a given list of values to retain a specified percentage that conforms
    best to a normal distribution.

    Parameters:
    - values (list or np.ndarray): The input data to filter.
    - retain_ratio (float): The percentage of values to retain (default is 0.8).
    - plot (bool): Whether to plot the filtered data and fitted normal distribution (default is False).

    Returns:
    - filtered_values (np.ndarray): The filtered subset of values.
    """
    values = np.array(values)

    # Step 1: Fit a normal distribution
    mean, std = np.mean(values), np.std(values)

    # Step 2: Calculate z-scores
    z_scores = np.abs((values - mean) / std)

    # Step 3: Sort values by z-scores
    sorted_indices = np.argsort(z_scores)
    sorted_values = values[sorted_indices]

    # Step 4: Retain the top percentage of values
    num_to_keep = int(retain_ratio * len(values))
    filtered_values = sorted_values[:num_to_keep]

    # Optional: Plot the filtered values and fitted normal distribution
    if plot:
        plot_dist(filtered_values)

    return filtered_values




def main():
    feature_sets = ['D_ECFP_4_MACCS_MFP_d1hot_mol_graph_smiles_C_c1hot']
    split_types = ["leave_comb", "leave_drug", "leave_cell_line"]
    score_names = ["S_mean_mean", "synergy_loewe_mean"]
    print(f'split\trun_no\tdataset\tuncommon_triplets\tuncommon_drugs\tuncommon_cell\ttotal_triplets\ttotal_drugs\ttotal_cell\t')

    for feature_set in feature_sets:
        all_df_dict = {}
        scores_dict = {}
        for split_type in split_types:
            for run_no in range(5):
                train_dfs = {}
                test_dfs = {}
                for score_name in score_names:
                    # score_name = score.replace("k_0.05_", "")
                    # print(score)
                    split_dir = f"/home/grads/tasnina/Projects/SynVerse/inputs/splits/{feature_set}/k_0.05_{score_name}/{split_type}_0.2_0.25/run_{run_no}/"
                    all_data_file = split_dir + 'all.tsv'
                    test_file = split_dir + 'test.tsv'
                    train_val_file = split_dir+'all_train.tsv'
                    val_idx_file = split_dir + 'val.pkl'
                    train_idx_file = split_dir + 'train.pkl'
                    cell_2_idx_file = split_dir + 'cell_line_2_idx.tsv'
                    cell_2_idx_df = pd.read_csv(cell_2_idx_file, sep='\t')[['cell_line_name', 'idx']]
                    idx_2_cell = dict(zip( cell_2_idx_df['idx'], cell_2_idx_df['cell_line_name']))

                    all_df = pd.read_csv(all_data_file, sep='\t')
                    test_df = pd.read_csv(test_file, sep='\t')
                    train_val_df = pd.read_csv(train_val_file, sep='\t')

                    all_df_dict[score_name] = all_df
                    scores_dict[score_name] = all_df[score_name]
                    train_dfs[score_name]=train_val_df
                    test_dfs[score_name]=test_df


                    # with open(val_idx_file, 'rb') as f:
                    #     val_data = pickle.load(f)
                    # with open(train_idx_file, 'rb') as f:
                    #     train_data = pickle.load(f)

                    # train_df = train_val_df.iloc[train_data]


                    #score distribution
                    # plt.hist(train_df[score], label='train', color='blue', bins=50)
                    # # plt.title(f'train {score_name} run {run_no}')
                    # # plt.show()
                    # plt.hist(test_df[score],  label='test', color='red', bins=50)
                    # plt.title(f'train test {score} run {run_no}')
                    # plt.legend(loc='upper right')
                    # plt.show()
                    #
                    # #check if the distribution is normal
                    # stats.probplot(train_df[score_name], dist="norm", plot=plt)
                    # plt.ylim(-250, 250)  # replace min_y and max_y with your desired values
                    # plt.title(f'{score_name}_train_{run_no}')
                    # plt.show()
                    #
                    # stats.probplot(test_df[score_name], dist="norm", plot=plt)
                    # plt.ylim(-250, 250)  # replace min_y and max_y with your desired values
                    # plt.title(f'{score_name}_test_{run_no}')
                    # plt.show()
                    #
                    # print("Skewness in train:", skew(train_df[score_name]))
                    # print("Kurtosis in train :", kurtosis(train_df[score_name]))
                    #
                    # print("Skewness in test:", skew(test_df[score_name]))
                    # print("Kurtosis in test:", kurtosis(test_df[score_name]))
                    #
                    # # KS test against normal distribution
                    # ks_stat, ks_p_value = stats.kstest(train_df[score_name], 'norm')
                    # print(f"Train KS: {ks_stat}, p-value: {ks_p_value}")
                    # ks_stat, ks_p_value = stats.kstest(test_df[score_name], 'norm')
                    # print(f"Test KS: {ks_stat}, p-value: {ks_p_value}")


                # prefix = f'{split_type}\t{run_no}\tALL  '
                # compare_data(all_data_dfs[score_names[0]],all_data_dfs[score_names[1]], prefix)
                prefix = f'{split_type}\t{run_no}\tTRAIN'
                compare_data(train_dfs[score_names[0]],train_dfs[score_names[1]], prefix)
                prefix = f'{split_type}\t{run_no}\tTEST'
                compare_data(test_dfs[score_names[0]], test_dfs[score_names[1]], prefix)


        compare_score_distributions(scores_dict[score_names[0]],scores_dict[score_names[1]], score_names[0], score_names[1])

        normal_filtered_scores_dict = {}
        for score_name in score_names:
            normal_filtered_scores_dict [score_name] = filter_normal_conforming_data(scores_dict[score_name], retain_ratio=0.99, plot=False)

        compare_score_distributions(normal_filtered_scores_dict[score_names[0]],normal_filtered_scores_dict[score_names[1]], score_names[0], score_names[1])


main()