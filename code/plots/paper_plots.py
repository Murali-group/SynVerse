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
from plots.plot_utils import *
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

feature_filters = ['D_ECFP_4_MACCS_MFP_d1hot_mol_graph_smiles_C_c1hot',
                  'D_d1hot_target_C_c1hot', 'D_d1hot_C_c1hot_genex_genex_lincs_1000']

#rename the long model names containing features, preprocessing and encoder name to suitable name for plot.
model_name_mapping = {'d1hot_std_comp_True + c1hot_std_comp_True': 'One hot (AE)',
'MACCS + c1hot': 'MACCS', 'MACCS_std_comp_True + c1hot_std_comp_True': 'MACCS (AE)',
'MFP + c1hot': 'MFP', 'MFP_std_comp_True + c1hot_std_comp_True': 'MFP (AE)',
'ECFP_4 + c1hot': 'ECFP', 'ECFP_4_std_comp_True + c1hot_std_comp_True': 'ECFP (AE)',
'mol_graph_GCN + c1hot': 'Mol Graph (GCN)',
'smiles_Transformer + c1hot': 'SMILES (Transformer)',
'smiles_SPMM + c1hot': 'SMILES (SPMM)',
'smiles_kpgt + c1hot': 'SMILES (KPGT)',
'smiles_mole + c1hot': 'SMILES (MolE)',
'target + c1hot': 'Target', 'target_rwr + c1hot': 'Target (RWR)',
'target_std_comp_True + c1hot_std_comp_True': 'Target (AE)',
'd1hot + genex_std': 'Genex',
'd1hot + genex_lincs_1000_std': 'LINCS_1000'}

def scatter_plot_model_comparison_with_deepsynergy(filename):
    df = pd.read_csv(filename, sep='\t')[['Model name', 'Own', 'DeepSynergy']]
    # Create the scatter plot

    plt.figure(figsize=(8, 6))
    plt.scatter(df['Own'], df['DeepSynergy'], cmap='viridis')
    plt.xlim([0, 1])
    plt.ylim([0, 1])

    # Plot the diagonal line (y=x) to compare the performances
    plt.plot([0, 1], [0,1], 'r--', label='Equal Performance Line')

    # Annotate each point with the model name
    texts = []
    for i, row in df.iterrows():
        text = plt.text(row['Own'] + 0.01, row['DeepSynergy'], row['Model name'], fontsize=9)
        texts.append(text)

    # Adjust the text positions to avoid overlap using adjustText
    adjust_text(texts)
    # Add labels and title
    plt.xlabel('Model Own Performance')
    plt.ylabel('DeepSynergy Performance')
    plt.title('Comparison of Model and DeepSynergy Performance')

    # Add a legend
    plt.legend()

    # Display the plot
    plt.tight_layout()
    plt.show()

def barplot_model_comparison_with_deepsynergy(filename):
    df = pd.read_csv(filename, sep='\t')[['Model name', 'Own', 'DeepSynergy', 'Threshold']]
    # Extracting the DeepSynergy's self-reported performance
    deep_synergy_self_performance = df.loc[df['Model name'].str.contains('DeepSynergy'), 'DeepSynergy'].iloc[0]

    # Setting up the plot
    df = df[df['Model name'].str.contains('DeepSynergy', case=False, na=False) == False]
    x = range(len(df))
    bar_width = 0.32

    fig, ax = plt.subplots(figsize=(8, 6))

    # Adding background colors based on 'Threshold'
    for idx, threshold in enumerate(df['Threshold']):
        if threshold == 10:
            ax.axvspan(idx - 0.5, idx + 0.5, facecolor='gainsboro', edgecolor='gainsboro')
        elif threshold == 30:
            ax.axvspan(idx - 0.5, idx + 0.5, facecolor='white', alpha=0.3, edgecolor='none')
        else:
            ax.axvspan(idx - 0.5, idx + 0.5, facecolor='lightgreen', alpha=0.3, edgecolor='none')

    colors = ListedColormap(cm.get_cmap('Paired').colors[:4])
    bars1 = ax.bar([i - bar_width / 2 for i in x], df['DeepSynergy'], bar_width,
                   label='DeepSynergy', color=colors(2))
    bars2 = ax.bar([i + bar_width / 2 for i in x], df['Own'], bar_width, label='Compared Model', color=colors(1))


    # Adding the line
    ax.axhline(y=deep_synergy_self_performance, color='red', linestyle='--',
               # label=f'Self-Reported AUPRC by DeepSynergy({deep_synergy_self_performance:.2f})'
               )

    # Adding labels and legend
    ax.set_xticks(x)
    # ax.set_xticklabels(df['Model name'])
    ax.set_xticklabels([name.replace(' (', '\n(') for name in df['Model name']])
    ax.set_ylabel('AUPRC')
    # ax.set_title('Model Performance Comparison')
    ax.legend()

    # Showing the plot
    ax.set_xlim(-0.5, len(df) - 0.5)
    plt.tight_layout()
    plt.savefig(os.path.dirname(filename) + '/barplot_model_comparison_with_deepsynergy.pdf')
    plt.show()

def compute_difference_with_1hot(df):
    baseline_df = df[(df['drug_features'] == 'd1hot') & (df['cell_features'] == 'c1hot')]
    #remove baseline_df from df
    df = df[~((df['drug_features'] == 'd1hot') & (df['cell_features'] == 'c1hot'))]
    # Merge the dataframe with itself on 'run_no' and 'feature_filter' to find pairs
    merged_df = df.merge(
        baseline_df,
        on=['run_no', 'feature_filter'],
        suffixes=('', '_baseline'),
        how='inner'
    )
    # Calculate differences for the required columns
    for column in ['test_loss', 'val_loss', 'train_loss']:
        merged_df[f'{column}_diff'] = merged_df[f'{column}_baseline']-merged_df[column]

    return merged_df





# def compute_avg_performance(df, n_runs=5):
#     # compute average and standard deviation of 'test_loss', 'val_loss', 'train_loss' along with
#     # average of difference with 1hot.
#
#     # Group by and compute required metrics
#     result = df.groupby(['drug_features', 'cell_features', 'feature_filter']).agg(
#         test_loss_mean=('test_loss', 'mean'),
#         test_loss_std=('test_loss', 'std'),
#         val_loss_mean=('val_loss', 'mean'),
#         val_loss_std=('val_loss', 'std'),
#         train_loss_mean=('train_loss', 'mean'),
#         train_loss_std=('train_loss', 'std'),
#         test_loss_diff_mean=('test_loss_diff', 'mean'),
#         val_loss_diff_mean=('val_loss_diff', 'mean'),
#         train_loss_diff_mean=('train_loss_diff', 'mean'),
#     ).reset_index()
#     #compute confidence interval for each model
#     result['test_loss_CI'] = result['test_loss_std'].astype(float).apply(lambda x: confidence_interval(x,n_runs, confidence_level=0.95))
#     result['val_loss_CI'] = result['val_loss_std'].astype(float).apply(lambda x: confidence_interval(x,n_runs, confidence_level=0.95))
#     result['train_loss_CI'] = result['train_loss_std'].astype(float).apply(lambda x: confidence_interval(x,n_runs, confidence_level=0.95))
#     return result

# def compute_diff_significance(df_1hot_diff):
#     # Define a function to compute the Mann-Whitney U test for a group
#     def compute_mannwhitney(group):
#         stat, p_value = mannwhitneyu(group['test_loss'], group['test_loss_baseline'], alternative='greater')
#         return pd.Series({'stat': stat, 'p_value': p_value})
#
#     # Group by the relevant columns and apply the test
#     results = (
#         df_1hot_diff
#         .groupby(['drug_features', 'celll_features', 'feature_filter'])
#         .apply(compute_mannwhitney)
#         .reset_index()
#     )
#
#     # Apply multiple testing correction (e.g., Benjamini-Hochberg)
#     adjusted_results = multipletests(results['p_value'], method='fdr_bh')  # fdr_bh for Benjamini-Hochberg
#     results['adjusted_p_value'] = adjusted_results[1]  # Adjusted p-values
#     results['is_significant'] = results['adjusted_p_value'] < 0.05  # Threshold for significance

def compute_average_and_significance(df, n_runs=5):
    # Define a function to compute the Mann-Whitney U test for a group
    def compute_mannwhitney(group):
        stat, p_value = mannwhitneyu(group['test_loss'], group['test_loss_baseline'], alternative='less')
        return pd.Series({'stat': stat, 'p_value': p_value})

    # Group by and compute aggregation metrics
    aggregated_results = df.groupby(['drug_features', 'cell_features', 'feature_filter']).agg(
        test_loss_mean=('test_loss', 'mean'),
        test_loss_max=('test_loss', 'max'),
        test_loss_std=('test_loss', 'std'),
        val_loss_mean=('val_loss', 'mean'),
        val_loss_max=('val_loss', 'max'),
        val_loss_std=('val_loss', 'std'),
        train_loss_mean=('train_loss', 'mean'),
        train_loss_max=('train_loss', 'max'),
        train_loss_std=('train_loss', 'std'),
        test_loss_diff_mean=('test_loss_diff', 'mean'),
        val_loss_diff_mean=('val_loss_diff', 'mean'),
        train_loss_diff_mean=('train_loss_diff', 'mean'),
    ).reset_index()

    # Compute confidence intervals for test, val, and train losses
    for loss_type in ['test_loss', 'val_loss', 'train_loss']:
        aggregated_results[f'{loss_type}_CI'] = aggregated_results[f'{loss_type}_std'].apply(
            lambda x: confidence_interval(x, n_runs, confidence_level=0.95)
        )

    # Compute significance of the test_loss compared to test_loss_baseline
    significance_results = (
        df.groupby(['drug_features', 'cell_features', 'feature_filter'])
        .apply(compute_mannwhitney)
        .reset_index()
    )

    # Apply multiple testing correction
    adjusted_results = multipletests(significance_results['p_value'], method='fdr_bh')
    significance_results['adjusted_p_value'] = adjusted_results[1]
    significance_results['is_significant'] = significance_results['adjusted_p_value'] < 0.05

    # Merge aggregated metrics and significance results
    final_results = pd.merge(
        aggregated_results,
        significance_results,
        on=['drug_features', 'cell_features', 'feature_filter']
    )

    return final_results
def plot_diff(df_1hot_diff_avg, y_label, title, metric='test_loss', yerr ='std', color_on = 'is_significant', out_file_prefix=None):
    """
    Plot a barplot with horizontally grouped subplots for each feature_filter with dynamic widths.
    The colorbar is positioned to the right of all subplots.

    Parameters:
    - df_1hot_diff_avg: DataFrame containing the required metrics.
    - metric: The metric to plot ('test_loss' by default).
    - title_prefix: Optional title prefix.
    """

    # Define columns of interest
    mean_col = f'{metric}_mean'
    err_col = f'{metric}_{yerr}'
    diff_mean_col = f'{metric}_diff_mean'

    if color_on != 'is_significant':
        color_on = diff_mean_col

    if color_on != 'is_significant':
        # Create a diverging color palette based on the diff_mean_col with 0 as the center
        abs_max_diff = max(abs(df_1hot_diff_avg[diff_mean_col].min()), abs(df_1hot_diff_avg[diff_mean_col].max()))
        norm = TwoSlopeNorm(vmin=-abs_max_diff, vcenter=0, vmax=abs_max_diff)
        cmap = sns.diverging_palette(220, 15, s=85, l=65, as_cmap=True).reversed()  # Colorblind-friendly palette
    else:
        colors = ['#ff8080', '#b3d7ff']
        cmap = ListedColormap(colors)

    # Get unique feature filters
    assert len(set(df_1hot_diff_avg['feature_filter'].unique()).difference(set(feature_filters)))==0, print('mismatch')

    # Create subplots with dynamic widths based on the number of rows in each subset
    row_counts = [len(df_1hot_diff_avg[df_1hot_diff_avg['feature_filter'] == feature_filter]) for feature_filter in feature_filters]
    total_rows = sum(row_counts)
    widths = [row_count / total_rows for row_count in row_counts]

    # Create subplots with dynamic widths
    fig, axes = plt.subplots(1, len(feature_filters),
                             figsize=(6 * len(feature_filters), 8),
                             gridspec_kw={'width_ratios': widths})
    if len(feature_filters) == 1:  # Ensure axes is always iterable
        axes = [axes]

    # # Calculate global y-axis limits
    # y_min = min(df_1hot_diff_avg[mean_col] - df_1hot_diff_avg[err_col])
    # y_max = max(df_1hot_diff_avg[mean_col] + df_1hot_diff_avg[err_col])
    #
    # # Optional: Adjust limits slightly for better aesthetics
    # padding = 0.01 * (y_max - y_min)
    # y_min -= padding
    # y_max += padding

    # Plot each feature_filter's data
    for ax, feature_filter in zip(axes, feature_filters):
        subset = df_1hot_diff_avg[df_1hot_diff_avg['feature_filter'] == feature_filter]

        if len(subset)==0:
            continue

        if color_on != 'is_significant':
            # Normalize colors for the subset
            colors = cmap(norm(subset[diff_mean_col]))
        else:
            colors = ['#ff8080', '#b3d7ff']
            cmap = ListedColormap(colors)
            colors = [cmap(value) for value in subset['is_significant']]

        # Plot bars
        bars = ax.bar(
            subset['Model'],
            subset[mean_col],
            yerr=subset[err_col],
            # yerr=subset[CI_col],
            color=colors,
            capsize=5,
            edgecolor='black'
        )
        # Calculate global y-axis limits
        y_min = min(subset[mean_col] - subset[err_col])
        y_max = max(subset[mean_col] + subset[err_col])

        # Optional: Adjust limits slightly for better aesthetics
        padding = 0.01 * (y_max - y_min)
        y_min -= padding
        y_max += padding

        ax.set_ylim(y_min, y_max)


        # Set x-tick labels for each subplot
        x_positions = [bar.get_x() + bar.get_width() / 2 for bar in bars]
        # print(x_positions)
        ax.set_xticks(x_positions)  # Set x-tick positions
        ax.set_xticklabels(subset['Model'], fontsize=14)  # Set x-tick labels
        ax.tick_params(axis='x', rotation=90)  # Rotate labels for better readability

        # Set y-axis label (Mean Squared Error) on the left side of each subplot
        # ax.set_ylabel("Mean Squared Error (MSE)", fontsize=12)

    fig.text(0.08, 0.5, y_label, va='center', rotation='vertical', fontsize=16)
    fig.text(0.5, -0.22, 'Models', ha='center', fontsize=16)


    if color_on !='is_significant':
        # # Create colorbar outside the subplots grid
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array(df_1hot_diff_avg[diff_mean_col])

        # Add a vertical colorbar to the right of the subplots
        cbar = fig.colorbar(sm, ax=axes, orientation='vertical', fraction=0.05, pad=0.04)
        cbar.set_label(f'Difference with baseline', fontsize=14)

    # # Adjust layout to ensure that the colorbar does not overlap the plots
    plt.subplots_adjust(right=0.85, wspace=0.15)  # Adjust right margin to create space for colorbar
    # fig.suptitle(f'{title}_{yerr}', fontsize=16, y=0.98)
    # Save the figure with tight bounding box (to avoid clipping)
    os.makedirs(os.path.dirname(out_file_prefix), exist_ok=True)
    plot_file = f"{out_file_prefix}_{metric}_{color_on}_plot.pdf"
    plt.savefig(plot_file, bbox_inches='tight')
    print(f'saved file: {plot_file}')
    # Show the plot
    plt.show()



def wrapper_plot_compare_with_1hot(df_MSE, title, out_file_prefix):

    #compute RMSE from MSE:
    df_RMSE= copy.deepcopy(df_MSE)
    for  metric in ['test_loss', 'train_loss','val_loss']:
        df_RMSE[metric] = np.sqrt(df_MSE[metric])

    data_dict = {'Mean Squared Error (MSE)': df_MSE, 'Root Mean Squared Error (RMSE)': df_RMSE}

    #compute the difference between the model's result and corresponding 1 hot encoding
    #plot MSE
    for measure in data_dict:
        file_name_suitable_measure = measure.split(' ')[-1].replace('(','').replace(')','')

        df_1hot_diff = compute_difference_with_1hot(data_dict[measure])
        df_1hot_diff_avg = compute_average_and_significance(df_1hot_diff)
        #remove a few feature combo if present
        df_1hot_diff_avg = df_1hot_diff_avg[~((df_1hot_diff_avg['drug_features']=='d1hot_comp_True')|(df_1hot_diff_avg['cell_features']=='c1hot_comp_True'))]

        # Create x-tick labels combining 'drug_features' and 'cell_features'
        df_1hot_diff_avg['Model'] = df_1hot_diff_avg['drug_features'] + " + " + df_1hot_diff_avg['cell_features']
        df_1hot_diff_avg['Model'] = df_1hot_diff_avg['Model'].astype(str).apply(lambda x: model_name_mapping.get(x, x))
        # Sort DataFrame according to the order of models in model_name_mapping
        df_1hot_diff_avg['Model'] = pd.Categorical(df_1hot_diff_avg['Model'], categories=model_name_mapping.values(),ordered=True)
        df_1hot_diff_avg = df_1hot_diff_avg.sort_values('Model')
        # remove model 'One hot (AE)'
        df_1hot_diff_avg = df_1hot_diff_avg[df_1hot_diff_avg['Model'] != 'One hot (AE)']
        df_1hot_diff_avg.to_csv(f'{out_file_prefix}_{file_name_suitable_measure}_aggreagred_performance.tsv', sep='\t')

        plot_diff(df_1hot_diff_avg, measure, title, metric ='test_loss', yerr='std', color_on = 'diff_mean', out_file_prefix=f'{out_file_prefix}_{file_name_suitable_measure}')
        plot_diff(df_1hot_diff_avg, measure, title, metric ='test_loss', yerr='std', color_on = 'is_significant', out_file_prefix=f'{out_file_prefix}_{file_name_suitable_measure}')


    print(title)



def main():
    # result_dir = '/home/grads/tasnina/Projects/SynVerse/outputs/k_0.05_S_mean_mean'
    result_dir = '/home/grads/tasnina/Projects/SynVerse/outputs/k_0.05_synergy_loewe_mean'
    score_name = result_dir.split('/')[-1].replace('k_0.05_','')

    split_types = ['random', 'leave_comb', 'leave_drug','leave_cell_line']
    for split_type in split_types:
        file_name = f'output_{split_type}.tsv'
        file_path = os.path.join(result_dir, file_name)
        if not os.path.exists(file_path):
            print(f'file {file_name} does not exist. Continuing to next file.')
            continue
        result_df = pd.read_csv(file_path, sep='\t', index_col=None)
        wrapper_plot_compare_with_1hot(result_df, title=split_type, out_file_prefix = f'{result_dir}/{score_name}_{split_type}')

    # barplot_model_comparison_with_deepsynergy("/home/grads/tasnina/Projects/SynVerse/inputs/existing_model_performance.tsv")
    # scatter_plot_model_comparison_with_deepsynergy("/home/grads/tasnina/Projects/SynVerse/inputs/existing_model_performance.tsv")


if __name__ == '__main__':
    main()