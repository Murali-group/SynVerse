import pickle
import pandas as pd
import matplotlib.pyplot as plt

def find_cell_lines_across_splits(train_df, val_df, test_df):
    val_cell_lines = val_df['cell_line_name'].unique()
    train_cell_lines = train_df['cell_line_name'].unique()
    test_cell_lines = test_df['cell_line_name'].unique()

    print('validation:', val_cell_lines)
    print('train:', train_cell_lines)
    print('test:', test_cell_lines)

def plot_cell_line_wise_score(prediction_file_of_interest, idx_2_cell):
    pred_df = pd.read_csv(prediction_file_of_interest, sep='\t')
    pred_df['cell_line_name'] = pred_df['cell_line'].apply(lambda x: idx_2_cell[x])

    plt.hist(pred_df['predicted'])
    plt.title(f'predicted score')
    plt.show()

    pred_df['squared_error'] = (pred_df['true']-pred_df['predicted'])**2

    # Group by cell line name and calculate mean squared error
    mse_df = pred_df.groupby('cell_line_name')['squared_error'].mean().reset_index()
    mse_df = mse_df.rename(columns={'squared_error': 'mean_squared_error'})

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.bar(mse_df['cell_line_name'], mse_df['mean_squared_error'])
    plt.xticks(rotation=45)
    plt.xlabel('Cell Line Name')
    plt.ylabel('Mean Squared Error')
    plt.title('Mean Squared Error by Cell Line')
    plt.tight_layout()
    plt.show()




def main():
    run_no=2

    split_dir = f"/home/grads/tasnina/Projects/SynVerse/inputs/splits/D_d1hot_C_c1hot_genex_genex_lincs_1000/k_0.05_S_mean_mean/leave_cell_line_0.2_0.25/run_{run_no}/"
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

    val_df = train_val_df.iloc[val_data]
    train_df = train_val_df.iloc[train_data]

    plt.hist(train_df['S_mean_mean'])
    plt.title(f'train score_run_{run_no}')
    plt.show()

    plt.hist(val_df['S_mean_mean'])
    plt.title(f'val score_run_{run_no}')
    plt.show()

    plt.hist(test_df['S_mean_mean'])
    plt.title(f'test score_run_{run_no}')
    plt.show()


    find_cell_lines_across_splits(train_df, val_df, test_df)

    prediction_file_of_interest = f'/home/grads/tasnina/Projects/SynVerse/outputs/k_0.05_S_mean_mean/leave_cell_line/run_{run_no}/D_d1hot_C_genex_lincs_1000_std_val_true_test_predicted_scores.tsv'
    plot_cell_line_wise_score(prediction_file_of_interest, idx_2_cell)


main()