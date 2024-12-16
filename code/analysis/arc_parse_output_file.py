import pandas as pd
import os
import re
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import matplotlib.pyplot as plt
import numpy as np

def plot_outputs(file_path, split_type):

    # Load the dataset
    df = pd.read_csv(file_path, sep='\t')
    # Grouping columns and metrics
    group_columns = ['drug_features', 'cell_features', 'one_hot_version']
    metrics = {
        'Training Loss': 'train_loss',
        'Test Loss': 'test_loss',
        'Precision_0': 'Precision_0',
        'Recall_0': 'Recall_0'
    }
    # Generate plots for each metric
    for metric_name, metric_col in metrics.items():
        grouped = df.groupby(group_columns)[metric_col].agg(['mean', 'std']).reset_index()

        # Create the bar plot
        fig, ax = plt.subplots(figsize=(10, 6))
        x_labels = grouped[group_columns].agg('-'.join, axis=1)
        x = np.arange(len(x_labels))
        ax.bar(x, grouped['mean'], yerr=grouped['std'], capsize=5)

        # Customize the plot
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=45, ha='right')
        ax.set_ylabel(metric_name)
        ax.set_title(f'{split_type}:{metric_name} Across Groups')
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()

        # Show the plot
        plt.show()


def compute_cls_performance(pred_file_path, thresholds = [0]):
    #columns = drug1	drug2	cell_line	TRUE	predicted
    pred_df = pd.read_csv(pred_file_path, sep='\t')
    prec_dict = {}
    rec_dict = {}
    for threshold in thresholds:
        y_true = list(pred_df['true'].astype(float).apply(lambda x : 1 if x > threshold else 0))
        y_pred = list(pred_df['predicted'].astype(float).apply(lambda x : 1 if x > threshold else 0))
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        prec_dict[f'Precision_{threshold}'] = precision
        rec_dict[f'Recall_{threshold}'] = recall
    return prec_dict, rec_dict

def read_loss_file_content(file_path):
    with open(file_path, 'r') as file:
        content = file.read()

        # Extract required information using regular expressions
        best_config_match = re.search(r"Best config: (.+)", content)
        epochs_match = re.search(r"Number of epochs: (\d+)", content)
        train_loss_match = re.search(r"train_loss: ([\d.]+)", content)
        test_loss_match = re.search(r"test_loss: ([\d.]+)", content)
        val_loss_match = re.search(r"val_loss: ([\d.]+)", content)


        # Get values if matches are found, else set to None
        best_config = eval(best_config_match.group(1)) if best_config_match else None
        epochs = int(epochs_match.group(1)) if epochs_match else None
        train_loss = float(train_loss_match.group(1)) if train_loss_match else None
        test_loss = float(test_loss_match.group(1)) if test_loss_match else None
        val_loss = float(val_loss_match.group(1)) if val_loss_match else None


        loss_dict = {'test_loss': test_loss,
            'val_loss': val_loss,
            'train_loss': train_loss,
            'num_epochs': epochs,
            'best_config': best_config}
    return loss_dict

def get_run_feat_info(file_path, run_number, one_hot_version=""):
    clean_file_name = file_path.split('/')[-1].replace("_val_true_loss.txt", "")
    features = re.sub(r'run_[0-4]', '', clean_file_name)  # Remove 'run_x' pattern
    drug_features = features.split('_C_')[0].replace('D_', '')
    cell_features = features.split('_C_')[1]
    run_info = {
        'run_no': run_number,
        'drug_features': drug_features,
        'cell_features': cell_features,
        'one_hot_version': one_hot_version,
    }
    return run_info

def iterate_output_files(folder_path):
    out_file_list = []
    # Iterate over each 'run_x' folder
    for run_folder in os.listdir(folder_path):
        run_path = os.path.join(folder_path, run_folder)

        # Ensure it's a directory and follows the 'run_x' pattern
        if os.path.isdir(run_path) and re.match(r'run_\d+', run_folder):
            run_number = run_folder  # Save run_x as run number
            # Iterate over each file in the 'run_x' folder
            for file_or_dirname in os.listdir(run_path):
                #************* REGRESSION LOSS *********************************
                # Consider files ending with '_loss.txt' to get train and test loss
                if file_or_dirname.endswith('_val_true_loss.txt'):
                    # Open and read the file content
                    loss_file_path = os.path.join(run_path, file_or_dirname)
                    pred_file_path = loss_file_path.replace('_val_true_loss.txt', '_val_true_test_predicted_scores.tsv')
                    run_info_dict = get_run_feat_info(loss_file_path, run_number, one_hot_version="-")
                    run_info_dict.update({'loss_file':loss_file_path, "pred_file": pred_file_path})
                    out_file_list.append(run_info_dict)

                # read files for one-hot encoding
                if 'One-hot-versions' in file_or_dirname:
                    sub_dirs = os.listdir(os.path.join(run_path, file_or_dirname))
                    for sub_dir in sub_dirs:
                        one_hot_files = os.listdir(os.path.join(run_path, file_or_dirname, sub_dir))
                        for one_hot_file in one_hot_files:
                            if one_hot_file.endswith('_val_true_loss.txt'):
                                # Open and read the file content
                                loss_file_path= os.path.join(run_path, file_or_dirname, sub_dir, one_hot_file)
                                pred_file_path = loss_file_path.replace('_val_true_loss.txt', '_val_true_test_predicted_scores.tsv')
                                run_info_dict = get_run_feat_info(loss_file_path, run_number, one_hot_version=sub_dir)
                                run_info_dict.update({'loss_file': loss_file_path, "pred_file": pred_file_path})
                                out_file_list.append(run_info_dict)
    # Create a DataFrame from the collected data
    return out_file_list



def main():
    # Example usage
    base_folder = '/home/tasnina/Projects/SynVerse/outputs/k_0.05_S_mean_mean/'
    split_types = ['leave_comb', 'leave_drug', 'leave_cell_line']
    outfile_detailed = base_folder + f'combined_output.xlsx'

    #extract output for features other than where both drug and cell lines have one-hot encoding
    #save outputs for each runs across all splits and features
    splitwise_df_dict = {}
    for split_type in split_types:
        splitwise_summary_file = base_folder+f'output_{split_type}.tsv'
        # plot_outputs(splitwise_summary_file)

        spec_folder = f'{base_folder}/{split_type}/'
        out_info_list = iterate_output_files(spec_folder)
        data = []
        for out_info in out_info_list:
            all_info = out_info
            loss_info = read_loss_file_content(out_info['loss_file'])
            precision, recall = compute_cls_performance(out_info['pred_file'], thresholds = [0, 10, 30])
            all_info.update(loss_info)
            all_info.update(precision)
            all_info.update(recall)
            data.append(all_info)
        df = pd.DataFrame(data)
        df.drop(columns=['loss_file', 'pred_file'], axis=1, inplace=True)
        splitwise_df_dict[split_type] = df
        print(df)
        df.to_csv(splitwise_summary_file, sep='\t', index=False)
        plot_outputs(splitwise_summary_file, split_type)


    with pd.ExcelWriter(outfile_detailed, mode="w") as writer:
        for split_type in splitwise_df_dict:
            splitwise_df_dict[split_type].to_excel(writer, sheet_name=split_type, index=False)




main()
