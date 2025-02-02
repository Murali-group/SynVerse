import copy
import os.path
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from adjustText import adjust_text
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
from matplotlib.colors import TwoSlopeNorm
from plot_utils import *
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
def main():
    out_dir = "/home/grads/tasnina/Projects/SynVerse/outputs"
    score_names = ['S_mean_mean', 'synergy_loewe_mean']
    split_types = ['leave_comb', 'leave_drug', 'leave_cell_line', 'random']
    retain_ratios = [0.99, 0.95]
    for score_name in score_names:
        orig_result_dir = f'{out_dir}/k_0.05_{score_name}'
        for split_type in split_types:
            df = pd.DataFrame()
            # plot for comparing models with each other. Also compare with one hot based model i.e., basleine
            result_file = f'output_{split_type}.tsv'
            for retain_ratio in retain_ratios:
                sampled_result_dir = f'{out_dir}/sample_norm_{retain_ratio}/k_0.05_{score_name}'
                sampled_file_path = os.path.join(sampled_result_dir, result_file)
                if not os.path.exists(sampled_file_path):
                    print(f'file {sampled_file_path} does not exist. Continuing to next file.')
                    continue
                sampled_result_df = pd.read_csv(sampled_file_path, sep='\t', index_col=None)
                sampled_result_df['sampled'] = retain_ratio
                df = pd.concat([df, sampled_result_df])

            if df.empty:
                continue

            orig_result_path = os.path.join(orig_result_dir, result_file)
            if not os.path.exists(orig_result_path):
                print(f'file {orig_result_path} does not exist. Continuing to next file.')
                continue

            orig_result_df = pd.read_csv(orig_result_path, sep='\t', index_col=None)
            orig_result_df['sampled'] = 1.0
            df = pd.concat([df, orig_result_df])
            df['test_loss_RMSE'] = np.sqrt(df['test_loss'])
            df = set_model_names(df)

            # keeps models  which I ran on sampled triplets
            sampled_model_names = df[df['sampled']<1]['Model'].unique()
            df = df[df['Model'].isin(sampled_model_names)]

            # sort model names
            df['Model'] = pd.Categorical(df['Model'], categories=model_name_mapping.values(), ordered=True)
            df = df.sort_values('Model')

            # modify model name to look good on plot
            df['Model'] = df['Model'].str.replace(r'\(', r'\n(', regex=True)

            out_file_prefix = out_dir + '/'+ score_name + '_' + split_type + '_subsampled'
            box_plot(df, x='Model', y='test_loss_RMSE', hue='sampled', ylabel='RMSE', rotate=90, palette="Set2",
                     out_file_prefix=out_file_prefix, title=split_type)


main()

