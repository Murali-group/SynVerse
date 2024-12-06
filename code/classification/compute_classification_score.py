import pandas as pd
import os
import re

def read_content( file_path, run_number, data, one_hot_version=''):
    with open(file_path, 'r') as file:
        content = file.read()

        # Extract required information using regular expressions
        best_config_match = re.search(r"Best config: (.+)", content)
        epochs_match = re.search(r"Number of epochs: (\d+)", content)
        train_loss_match = re.search(r"train_loss: ([\d.]+)", content)
        test_loss_match = re.search(r"test_loss: ([\d.]+)", content)

        # Get values if matches are found, else set to None
        best_config = eval(best_config_match.group(1)) if best_config_match else None
        epochs = int(epochs_match.group(1)) if epochs_match else None
        train_loss = float(train_loss_match.group(1)) if train_loss_match else None
        test_loss = float(test_loss_match.group(1)) if test_loss_match else None

        clean_file_name = file_path.split('/')[-1].replace("_loss.txt", "")
        features = re.sub(r'run_[0-4]', '', clean_file_name)  # Remove 'run_x' pattern

        drug_features = features.split('_C_')[0].replace('D_', '')
        cell_features = features.split('_C_')[1]
        # Append data as a row in the list
        data.append({
            'run_no': run_number,
            'drug_features': drug_features,
            'cell_features': cell_features,
            'test_loss': test_loss,
            'train_loss': train_loss,
            'num_epochs': epochs,
            'one_hot_version': one_hot_version,
            'best_config': best_config,
        })
    return data

def iterate_files(folder_path):
    # Iterate over each 'run_x' folder
    for run_folder in os.listdir(folder_path):
        run_path = os.path.join(folder_path, run_folder)

        # Ensure it's a directory and follows the 'run_x' pattern
        if os.path.isdir(run_path) and re.match(r'run_\d+', run_folder):
            run_number = run_folder  # Save run_x as run number

            # Iterate over each file in the 'run_x' folder
            for file_or_dirname in os.listdir(run_path):
                # Consider files ending with '_loss.txt' but not 'train_val_loss.txt'
                if file_or_dirname.endswith('_test_predicted_scores.tsv'):
                    # Open and read the file content
                    file_path = os.path.join(run_path, file_or_dirname)
                    scores_df = pd.read_csv(file_path, sep='\t')

                # read files for one-hot encoding
                if 'One-hot-versions' in file_or_dirname:
                    sub_dirs = os.listdir(os.path.join(run_path, file_or_dirname))
                    for sub_dir in sub_dirs:
                        one_hot_files = os.listdir(os.path.join(run_path, file_or_dirname, sub_dir))
                        for one_hot_file in one_hot_files:
                            if one_hot_file.endswith('_test_predicted_scores.tsv'):
                                # Open and read the file content
                                file_path= os.path.join(run_path, file_or_dirname, sub_dir, one_hot_file)
                                scores_df = pd.read_csv(file_path, sep='\t')



    # Create a DataFrame from the collected data
    df = pd.DataFrame(data)
    return df



def main():
    # Example usage
    base_folder = '/home/grads/tasnina/Projects/SynVerse/outputs/k_0.05/'
    split_types = ['leave_comb', 'leave_drug', 'leave_cell_line']
    outfile_detailed = base_folder + f'combined_output.xlsx'

    #extract output for features other than where both drug and cell lines have one-hot encoding
    #save outputs for each runs across all splits and features
    splitwise_df_dict = {}
    for split_type in split_types:
        spec_folder = f'{base_folder}/{split_type}/'
        df = iterate_files(spec_folder)
        splitwise_df_dict[split_type] = df
        print(df)

    with pd.ExcelWriter(outfile_detailed, mode="w") as writer:
        for split_type in splitwise_df_dict:
            splitwise_df_dict[split_type].to_excel(writer, sheet_name=split_type, index=False)




main()
