import pandas as pd
import os
import re
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import matplotlib.pyplot as plt
import numpy as np

def extract_and_save_best_config(loss_file_path):
    with open(loss_file_path, 'r') as file:
        content = file.read()

        # Extract required information using regular expressions
        best_config_match = re.search(r"Best config: (.+)", content)
        epochs_match = re.search(r"Number of epochs: (\d+)", content)

        # Get values if matches are found, else set to None
        best_config = eval(best_config_match.group(1)) if best_config_match else None
        epochs = int(epochs_match.group(1)) if epochs_match else None

    best_config_file = loss_file_path.replace('_loss.txt', '_best_hyperparam.txt')
    with open(best_config_file, 'w') as f:
        f.write('best_config: ' + str(best_config))
        f.write('\nbest_epochs: ' + str(int(epochs)))
    f.close()



def save_config_in_sep_file(folder_path):
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
                if file_or_dirname.endswith('_loss.txt'):
                    # Open and read the file content
                    loss_file_path = os.path.join(run_path, file_or_dirname)
                    extract_and_save_best_config(loss_file_path)


                # read files for one-hot encoding
                if 'One-hot-versions' in file_or_dirname:
                    sub_dirs = os.listdir(os.path.join(run_path, file_or_dirname))
                    for sub_dir in sub_dirs:
                        one_hot_files = os.listdir(os.path.join(run_path, file_or_dirname, sub_dir))
                        for one_hot_file in one_hot_files:
                            if one_hot_file.endswith('_loss.txt'):
                                # Open and read the file content
                                loss_file_path= os.path.join(run_path, file_or_dirname, sub_dir, one_hot_file)
                                extract_and_save_best_config(loss_file_path)

    # Create a DataFrame from the collected data
    return out_file_list



def main():
    # Example usage
    base_folder = '/home/grads/tasnina/Projects/SynVerse/outputs/k_0.05_S_mean_mean/'
    split_types = ['leave_comb', 'leave_drug', 'leave_cell_line']


    for split_type in split_types:
        spec_folder = f'{base_folder}/{split_type}/'
        out_info_list = save_config_in_sep_file(spec_folder)





main()
