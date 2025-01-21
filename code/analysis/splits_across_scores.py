import pickle
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import skew, kurtosis


def get_synergy_stat(df):
    drugs = set(df['drug_1_pid']).union(set(df['drug_2_pid']))
    cell_lines = set(df['cell_line_name'])
    triplets = set(zip(df['drug_1_pid'], df['drug_2_pid'], df['cell_line_name']))
    return triplets, drugs, cell_lines
def compare_data(dfs):
    '''compare the two df present in dfs.'''
    df1=dfs[0]
    df2=dfs[1]
    triplets1, drugs1, cell_lines1 = get_synergy_stat(df1)
    triplets2, drugs2, cell_lines2 = get_synergy_stat(df2)

    common_triplets = triplets1.intersection(triplets2)
    total_triplets = triplets1.union(triplets2)
    common_drugs = drugs1.intersection(drugs2)
    total_drugs = drugs1.union(drugs2)
    common_cell_lines = cell_lines1.intersection(cell_lines2)
    total_cell_lines = cell_lines1.union(cell_lines2)

    print('triplets: ', len(common_triplets), '  drugs: ',len(common_drugs), ' cell lines: ',len(common_cell_lines))
    print('triplets: ',len(total_triplets),'  drugs: ', len(total_drugs),' cell lines: ', len(total_cell_lines))


def main():
    scores = ["k_0.05_S_mean_mean", "k_0.05_synergy_loewe_mean"]
    for run_no in range(5):

        train_dfs = []
        test_dfs = []

        for score in scores:
            score_name = score.replace("k_0.05_", "")
            print(score)
            split_dir = f"/home/grads/tasnina/Projects/SynVerse/inputs/splits/D_ECFP_4_MACCS_MFP_d1hot_mol_graph_smiles_C_c1hot/{score}/leave_drug_0.2_0.25/run_{run_no}/"
            test_file = split_dir + 'test.tsv'
            train_val_file = split_dir+'all_train.tsv'
            val_idx_file = split_dir + 'val.pkl'
            train_idx_file = split_dir + 'train.pkl'
            cell_2_idx_file = split_dir + 'cell_line_2_idx.tsv'
            cell_2_idx_df = pd.read_csv(cell_2_idx_file, sep='\t')[['cell_line_name', 'idx']]
            idx_2_cell = dict(zip( cell_2_idx_df['idx'], cell_2_idx_df['cell_line_name']))

            test_df = pd.read_csv(test_file, sep='\t')
            train_val_df = pd.read_csv(train_val_file, sep='\t')

            with open(val_idx_file, 'rb') as f:
                val_data = pickle.load(f)
            with open(train_idx_file, 'rb') as f:
                train_data = pickle.load(f)

            train_df = train_val_df.iloc[train_data]
            train_dfs.append(train_df)
            test_dfs.append(test_df)

            #score distribution
            plt.hist(train_df[score_name], label='train', color='blue', bins=50)
            # plt.title(f'train {score_name} run {run_no}')
            # plt.show()
            plt.hist(test_df[score_name],  label='test', color='red', bins=50)
            plt.title(f'train test {score_name} run {run_no}')
            plt.legend(loc='upper right')
            plt.show()

            #check if the distribution is normal
            stats.probplot(train_df[score_name], dist="norm", plot=plt)
            plt.ylim(-250, 250)  # replace min_y and max_y with your desired values
            plt.title(f'{score_name}_train_{run_no}')
            plt.show()

            stats.probplot(test_df[score_name], dist="norm", plot=plt)
            plt.ylim(-250, 250)  # replace min_y and max_y with your desired values
            plt.title(f'{score_name}_test_{run_no}')
            plt.show()

            print("Skewness in train:", skew(train_df[score_name]))
            print("Kurtosis in train :", kurtosis(train_df[score_name]))

            print("Skewness in test:", skew(test_df[score_name]))
            print("Kurtosis in test:", kurtosis(test_df[score_name]))

            # KS test against normal distribution
            ks_stat, ks_p_value = stats.kstest(train_df[score_name], 'norm')
            print(f"Train KS: {ks_stat}, p-value: {ks_p_value}")
            ks_stat, ks_p_value = stats.kstest(test_df[score_name], 'norm')
            print(f"Test KS: {ks_stat}, p-value: {ks_p_value}")

        print('run no: ', run_no)
        print('compare train data: ')
        compare_data(train_dfs)
        print('compare test data: ')
        compare_data(test_dfs)

main()