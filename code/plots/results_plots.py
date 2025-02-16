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
import matplotlib.colors as mcolors


single_color = mcolors.to_rgba("#196f3d", alpha=0.4)
edge_color='#85929e'
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

    for column in ['test_MSE', 'val_MSE', 'train_MSE', 'test_RMSE', 'val_RMSE', 'train_RMSE',
                   'Pearsons', 'Spearman']:
        merged_df[f'{column}_diff'] =  merged_df[column]-merged_df[f'{column}_baseline']


    return merged_df




def compute_average_with_1hot_diff(df):
    aggregated_results = df.groupby(['drug_features', 'cell_features', 'feature_filter','Model']).agg(
        test_MSE_mean=('test_MSE', 'mean'),
        test_MSE_max=('test_MSE', 'max'),
        test_MSE_std=('test_MSE', 'std'),
        val_MSE_mean=('val_MSE', 'mean'),
        val_MSE_max=('val_MSE', 'max'),
        val_MSE_std=('val_MSE', 'std'),
        train_MSE_mean=('train_MSE', 'mean'),
        train_MSE_max=('train_MSE', 'max'),
        train_MSE_std=('train_MSE', 'std'),
        test_RMSE_mean=('test_RMSE', 'mean'),
        test_RMSE_max=('test_RMSE', 'max'),
        test_RMSE_std=('test_RMSE', 'std'),
        val_RMSE_mean=('val_RMSE', 'mean'),
        val_RMSE_max=('val_RMSE', 'max'),
        val_RMSE_std=('val_RMSE', 'std'),
        train_RMSE_mean=('train_RMSE', 'mean'),
        train_RMSE_max=('train_RMSE', 'max'),
        train_RMSE_std=('train_RMSE', 'std'),
        Pearsons_mean=('Pearsons', 'mean'),
        Pearsons_std=('Pearsons', 'std'),
        Pearsons_max=('Pearsons', 'max'),

        Spearman_mean=('Spearman', 'mean'),
        Spearman_std=('Spearman', 'std'),
        Spearman_max=('Spearman', 'max'),

        test_MSE_diff_mean=('test_MSE_diff', 'mean'),
        val_MSE_diff_mean=('val_MSE_diff', 'mean'),
        train_MSE_diff_mean=('train_MSE_diff', 'mean'),
        test_RMSE_diff_mean=('test_RMSE_diff', 'mean'),
        val_RMSE_diff_mean=('val_RMSE_diff', 'mean'),
        train_RMSE_diff_mean=('train_RMSE_diff', 'mean'),
        Pearsons_diff_mean=('Pearsons_diff', 'mean'),
        Spearman_diff_mean=('Spearman_diff', 'mean'),

    ).reset_index()
    return aggregated_results


def compute_average(df):
    aggregated_results = df.groupby(['drug_features', 'cell_features', 'feature_filter','Model']).agg(
        test_MSE_mean=('test_MSE', 'mean'),
        test_MSE_max=('test_MSE', 'max'),
        test_MSE_std=('test_MSE', 'std'),
        val_MSE_mean=('val_MSE', 'mean'),
        val_MSE_max=('val_MSE', 'max'),
        val_MSE_std=('val_MSE', 'std'),
        train_MSE_mean=('train_MSE', 'mean'),
        train_MSE_max=('train_MSE', 'max'),
        train_MSE_std=('train_MSE', 'std'),
        test_RMSE_mean=('test_RMSE', 'mean'),
        test_RMSE_max=('test_RMSE', 'max'),
        test_RMSE_std=('test_RMSE', 'std'),
        val_RMSE_mean=('val_RMSE', 'mean'),
        val_RMSE_max=('val_RMSE', 'max'),
        val_RMSE_std=('val_RMSE', 'std'),
        train_RMSE_mean=('train_RMSE', 'mean'),
        train_RMSE_max=('train_RMSE', 'max'),
        train_RMSE_std=('train_RMSE', 'std'),
        Pearsons_mean=('Pearsons', 'mean'),
        Pearsons_std=('Pearsons', 'std'),
        Pearsons_max=('Pearsons', 'max'),
        Spearman_mean=('Spearman', 'mean'),
        Spearman_std=('Spearman', 'std'),
        Spearman_max=('Spearman', 'max'),

    ).reset_index()

    # Restore categorical ordering for 'Model'
    aggregated_results['Model'] = pd.Categorical(aggregated_results['Model'], categories=model_name_mapping.values(), ordered=True)

    # Explicitly sort again based on categorical order
    aggregated_results = aggregated_results.sort_values('Model')
    return aggregated_results

def compute_average_and_significance(df, measure, n_runs=5):
    # Group by and compute aggregation metrics
    aggregated_results = compute_average_with_1hot_diff(df)

    # Define a function to compute the Mann-Whitney U test for a group
    def compute_mannwhitney(group):
        stat, p_value = mannwhitneyu(group[measure], group[f'{measure}_baseline'], alternative='less')
        return pd.Series({'stat': stat, 'p_value': p_value})





    # Compute significance of the test_loss compared to test_loss_baseline
    significance_results = (
        df.groupby(['drug_features', 'cell_features', 'feature_filter','Model'])
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
        on=['drug_features', 'cell_features', 'feature_filter','Model']
    )

    return final_results
def plot_diff(df_1hot_diff_avg, metric, y_label, yerr ='std', out_file_prefix=None):
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

    # Create a diverging color palette based on the diff_mean_col with 0 as the center
    abs_max_diff = max(abs(df_1hot_diff_avg[diff_mean_col].min()), abs(df_1hot_diff_avg[diff_mean_col].max()))
    norm = TwoSlopeNorm(vmin=-abs_max_diff, vcenter=0, vmax=abs_max_diff)
    if (metric == 'Pearsons') | (metric == 'Spearman'):
        cmap = sns.diverging_palette(220, 15, s=85, l=65, as_cmap=True).reversed()
    else:
        cmap = sns.diverging_palette(220, 15, s=85, l=65, as_cmap=True)


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

    if (metric == 'Pearsons')|(metric == 'Spearman'):
        y_max=1
        y_min = min(min(df_1hot_diff_avg[mean_col]-df_1hot_diff_avg[err_col]), 0)
    else:
        # # Calculate global y-axis limits. Make ylims divisible by 5
        y_max = max(df_1hot_diff_avg[mean_col] + df_1hot_diff_avg[err_col])
        # y_min = y_min-(y_min%5)-5
        y_min=0
        y_max = y_max-(y_max%5)+5

    # Optional: Adjust limits slightly for better aesthetics
    # padding = 0.01 * (y_max - y_min)
    # y_min -= padding
    # y_max += padding

    # Plot each feature_filter's data
    for ax, feature_filter in zip(axes, feature_filters):
        subset = df_1hot_diff_avg[df_1hot_diff_avg['feature_filter'] == feature_filter]

        if len(subset)==0:
            continue

        # Normalize colors for the subset
        colors = cmap(norm(subset[diff_mean_col]))


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
        # y_min = min(subset[mean_col] - subset[err_col])
        # y_max = max(subset[mean_col] + subset[err_col])
        #
        # # Optional: Adjust limits slightly for better aesthetics
        # padding = 0.01 * (y_max - y_min)
        # y_min -= padding
        # y_max += padding

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
    plot_file = f"{out_file_prefix}_with_baseline_barplot.pdf"
    plt.savefig(plot_file, bbox_inches='tight')
    print(f'saved file: {plot_file}')
    # Show the plot
    plt.show()


def plot_performance_subplots(df_avg, metric, y_label, title, yerr ='std', out_file_prefix=None):
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

    # Get unique feature filters
    assert len(set(df_avg['feature_filter'].unique()).difference(set(feature_filters))) == 0, print('mismatch')

    # Create subplots with dynamic widths based on the number of rows in each subset
    row_counts = [len(df_avg[df_avg['feature_filter'] == feature_filter]) for feature_filter in feature_filters]
    total_rows = sum(row_counts)
    widths = [row_count / total_rows for row_count in row_counts]

    # Create subplots with dynamic widths
    fig, axes = plt.subplots(1, len(feature_filters),
                             figsize=(3.5 * len(feature_filters), 6),
                             gridspec_kw={'width_ratios': widths})
    if len(feature_filters) == 1:  # Ensure axes is always iterable
        axes = [axes]

    if (metric == 'Pearsons')|(metric == 'Spearman'):
        y_max=1
        y_min = min(min(df_avg[mean_col]-df_avg[err_col]), 0)
    else:
        # # Calculate global y-axis limits. Make ylims divisible by 5
        y_max = max(df_avg[mean_col] + df_avg[err_col])
        # y_min = y_min-(y_min%5)-5
        y_min=0
        y_max = y_max-(y_max%5)+5


    #model colors
    # unique_models = df_avg['Model'].unique()
    # color_palette = sns.cubehelix_palette(start=0.3, hue=1,
    #                                   gamma=0.4, dark=0.1, light=0.8,
    #                                   rot=-1, reverse=False,  n_colors=len(unique_models))
    # model_colors = {model: color for model, color in zip(unique_models, color_palette)}

    # color_palette = sns.cubehelix_palette(start=0.3, hue=1,
    #                                   gamma=0.4, dark=0.4, light=0.8,
    #                                   rot=-0.7, reverse=False,  n_colors=len(feature_filters))
    # feature_filter_colors = {feature_filter: color for feature_filter, color in zip(feature_filters, color_palette)}

    # hatch_patterns = ['///', 'xxx', '---', '|', '+', 'o', '*', '.']  # Add more if needed
    # feature_filter_hatches = {feature_filter: hatch for feature_filter, hatch in zip(feature_filters, hatch_patterns)}

    # Plot each feature_filter's data
    for ax, feature_filter in zip(axes, feature_filters):
        subset = df_avg[df_avg['feature_filter'] == feature_filter]

        if len(subset)==0:
            continue

        # Plot bars
        bars = ax.bar(
            subset['Model'],
            subset[mean_col],
            yerr=subset[err_col],
            capsize=5,
            edgecolor=edge_color,
            width=0.65,
            # alpha=0.4,
            color = single_color,
            # color=[model_colors[model] for model in subset['Model']]
            # color=[feature_filter_colors[feature_filter]]
            # hatch=feature_filter_hatches[feature_filter]  # Apply different hatches

        )

        ax.set_ylim(y_min, y_max)
        ax.set_xlim(-0.5, len(subset['Model']) - 0.5)

        # Set x-tick labels for each subplot
        x_positions = [bar.get_x() + bar.get_width() / 2 for bar in bars]
        # print(x_positions)
        ax.set_xticks(x_positions)  # Set x-tick positions
        ax.set_xticklabels(subset['Model'], fontsize=14)  # Set x-tick labels
        ax.tick_params(axis='x', rotation=90)  # Rotate labels for better readability
        ax.yaxis.grid(True, linestyle='--', color='grey', alpha=0.6, linewidth=0.6)

        # Set y-axis label (Mean Squared Error) on the left side of each subplot
        # ax.set_ylabel("Mean Squared Error (MSE)", fontsize=12)

    fig.text(0.05, 0.5, y_label, va='center', rotation='vertical', fontsize=16)
    # fig.text(0.5, -0.22, 'Models', ha='center', fontsize=16)

    # fig.text(0.5, 0.95, title, ha='center', fontsize=16)



    # # Adjust layout to ensure that the colorbar does not overlap the plots
    # plt.subplots_adjust(right=0.85, wspace=0.2)  # Adjust right margin to create space for colorbar
    # fig.suptitle(f'{title}_{yerr}', fontsize=16, y=0.98)
    # Save the figure with tight bounding box (to avoid clipping)
    if out_file_prefix is not None:
        os.makedirs(os.path.dirname(out_file_prefix), exist_ok=True)
        plot_file = f"{out_file_prefix}_barplot.pdf"
        plt.savefig(plot_file, bbox_inches='tight')
        print(f'saved file: {plot_file}')
    # Show the plot
    plt.show()
    print('done')


def plot_performance(df_avg, metric, y_label, title, yerr='std', out_file_prefix=None):
    """
    Plot a barplot for the given metric, coloring each model distinctly.

    Parameters:
    - df_avg: DataFrame containing the required metrics.
    - metric: The metric to plot ('test_loss' by default).
    - y_label: Label for the y-axis.
    - title: Plot title.
    - yerr: Column name for error values.
    - out_file_prefix: Prefix for saving the output file.
    """
    mean_col = f'{metric}_mean'
    err_col = f'{metric}_{yerr}'

    assert len(
        set(df_avg['feature_filter'].unique()).difference(set(feature_filters))) == 0, "Mismatch in feature filters"

    fig, ax = plt.subplots(figsize=(len(df_avg['Model'].unique()), 8))

    if metric in ['Pearsons', 'Spearman']:
        y_max = 1
        y_min = min(min(df_avg[mean_col] - df_avg[err_col]), 0)
    else:
        y_max = max(df_avg[mean_col] + df_avg[err_col])
        y_min = 0
        y_max = y_max - (y_max % 5) + 5

    # Assign distinct colors to models
    unique_models = df_avg['Model'].unique()
    color_palette = sns.cubehelix_palette(start=0.3, hue=1,
                                          gamma=0.4, dark=0.1, light=0.8,
                                          rot=-1.5, reverse=False, n_colors=len(unique_models))
    model_colors = {model: color for model, color in zip(unique_models, color_palette)}
    edge_color = "#666666"
    edge_width = 0.5
    # Plot bars
    bars = ax.bar(
        df_avg['Model'],
        df_avg[mean_col],
        yerr=df_avg[err_col],
        capsize=5,
        edgecolor=edge_color,
        linewidth=edge_width,
        color=[model_colors[model] for model in df_avg['Model']]
    )
    ax.set_ylabel(y_label, fontsize=16)

    ax.set_ylim(y_min, y_max)
    ax.set_xticks(range(len(df_avg['Model'])))
    ax.set_xticklabels(df_avg['Model'], fontsize=14, rotation=90)

    # fig.text(0.08, 0.5, y_label, va='center', rotation='vertical', fontsize=16)
    # fig.text(0.5, -0.22, 'Models', ha='center', fontsize=16)
    # fig.text(0.5, 0.95, title, ha='center', fontsize=16)

    # Save the figure
    os.makedirs(os.path.dirname(out_file_prefix), exist_ok=True)
    plot_file = f"{out_file_prefix}_barplot.pdf"
    plt.savefig(plot_file, bbox_inches='tight')
    print(f'Saved file: {plot_file}')

    plt.show()
    print('Done')


def pair_plot(df_all, metric, out_file_prefix):

    models = df_all['Model'].unique()


    # Create a grid of 4x4 subplots
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))  # Adjust figure size as needed
    axes = axes.flatten()  # Flatten to easily iterate through axes

    max_loss = max(df_all[metric].max(), df_all[f'{metric}_baseline'].max())
    min_val = min (df_all[metric].min(), df_all[f'{metric}_baseline'].min(), 5)

    for (i,model) in enumerate(models):
        df = df_all[df_all['Model'] == model]
        ax = axes[i]
        # Create the pair plot
        # colors = ['red' if x > y else 'blue' for x, y in zip( df['test_loss'], df['test_loss_baseline'])]

        sns.scatterplot(
            x=metric,
            y=f'{metric}_baseline',
            data=df,
            s=50,
            # c=colors,
            alpha=0.7,
            ax=ax
        )

        # Add a diagonal line to represent equal test losses
        # max_loss = max(df['test_loss'].max(), df['test_loss_baseline'].max())
        # min_val=5
        ax.set_xlim(min_val, max_loss)
        ax.set_ylim(min_val, max_loss)

        ax.plot([min_val, max_loss], [min_val, max_loss], linestyle='--',alpha=0.7, linewidth=0.7, color='grey' )

        # Customize the plot
        ax.set_xlabel('')
        ax.set_ylabel('')

        ax.set_title(model, fontsize=10)

    # Add common x and y axis labels
    fig.text(0.5, 0.04, f'Model\'s {metric}', ha='center', fontsize=14)
    fig.text(0.04, 0.5, f'Baseline\'s {metric}', va='center', rotation='vertical', fontsize=14)

    # Turn off empty subplots if there are any
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Show and save the plot
    plt.tight_layout(rect=[0.05, 0.05, 1, 0.97])  # Adjust space for labels and title

    # plt.tight_layout()
    out_file = f'{out_file_prefix}_with_baseline_pairplot.pdf'
    plt.savefig(out_file, bbox_inches='tight')
    print(f'saving file at {out_file}')
    plt.show()



def wrapper_plot_model_performance(df, metric, y_label,title, out_file_prefix=None):

    df = compute_average(df)

    df.to_csv(f'{out_file_prefix}_aggreagred.tsv', sep='\t')
    # remove one-hot based model
    df = df[df['Model'] != 'One hot']

    #bar plot for showing Pearsons, RMSE or some other metric of each model, showing performance improvement over baseline with color.
    # plot_performance(df, metric=metric, y_label=y_label, title=title, yerr='std', out_file_prefix=f'{out_file_prefix}')
    plot_performance_subplots(df, metric=metric, y_label=y_label, title=title, yerr='std', out_file_prefix=out_file_prefix)


    print(title)

def wrapper_plot_compare_with_1hot(df, metric, y_label, title, out_file_prefix):
    df_1hot_diff = compute_difference_with_1hot(df)
    if df_1hot_diff.empty:
        return

    df_1hot_diff.to_csv(f'{out_file_prefix}_with_baseline.tsv', sep='\t')

    #pair plot for comparing each modelw ith baseline across each individual run
    pair_plot(df_1hot_diff, metric, out_file_prefix=f'{out_file_prefix}')

    if metric =='Pearsons':
        y_min=-0.1
        y_max=0.1
    else:
        y_min= None
        y_max = None
    unique_models = df_1hot_diff['Model'].unique()

    #for one color
    # remove one-hot based model
    df_1hot_diff = df_1hot_diff[df_1hot_diff['Model'] != 'One hot']


    box_plot(df_1hot_diff, x='Model', y=f'{metric}_diff', ylabel='Improvement over baseline (Pearson\'s)',
             rotate=90, y_min=y_min, y_max=y_max, color = single_color,
             width=0.7, dodge=False, zero_line=True,legend=False, out_file_prefix=f'{out_file_prefix}')

    print(title)




def wrapper_plot_compare_rewired(result_df, rewired_result_df, metric, y_label, out_file_prefix):

        df = pd.concat([result_df, rewired_result_df], axis=0)
        # df = set_model_names(df)


        #keeps models  which I ran on rewired network
        rewired_model_names = df[df['rewired']==True]['Model'].unique()
        df = df[df['Model'].isin(rewired_model_names)]

        # #sort model names
        # df['Model'] = pd.Categorical(df['Model'], categories=model_name_mapping.values(),ordered=True)
        # df = df.sort_values('Model')

        #modify model name to look good on plot
        # df['Model'] = df['Model'].str.replace(r'\(', r'\n(', regex=True)
        df_1 = df.groupby(['Model','rewire_method']).agg({'test_RMSE': 'mean', 'Pearsons': 'mean', 'Spearman': 'mean'})
        df_1.to_csv(f'{out_file_prefix}_aggregated.tsv', sep='\t')
        # print(df_1)
        if (metric == 'Pearsons') | (metric == 'Spearman'):
            y_max=1
            y_min = min(min(df[metric]), 0)
        else:
            y_max=None
            y_min=None
        unique_models = df['Model'].unique()

        box_plot(df, x='Model', y=metric, hue='rewire_method', ylabel=y_label,y_min=y_min, y_max=y_max, palette="Set2", rotate=90,
                 figsize=(len(unique_models), 8), out_file_prefix=out_file_prefix)


def wrapper_plot_compare_shuffled(result_df, shuffled_result_df, metric, y_label, out_file_prefix):
    df = pd.concat([result_df, shuffled_result_df], axis=0)
    # df = set_model_names(df)

    # keeps models  which I ran with shuffled features
    shuffled_model_names = df[df['shuffled'] == True]['Model'].unique()
    df = df[df['Model'].isin(shuffled_model_names)]

    # sort model names
    # df['Model'] = pd.Categorical(df['Model'], categories=model_name_mapping.values(), ordered=True)
    # df = df.sort_values('Model')

    # modify model name to look good on plot
    # df['Model'] = df['Model'].str.replace(r'\(', r'\n(', regex=True)
    df_1 = df.groupby(['Model', 'shuffle_method']).agg({'test_RMSE': 'mean', 'Pearsons': 'mean', 'Spearman': 'mean'})
    df_1.to_csv(f'{out_file_prefix}_aggregated.tsv', sep='\t')

    if (metric == 'Pearsons') | (metric == 'Spearman'):
        y_max = 1
        y_min = min(min(df[metric]), 0.5)
    else:
        y_max = None
        y_min = None
    # modify model name to look good on plot
    unique_models = df['Model'].unique()
    box_plot(df, x='Model', y=metric, hue='shuffle_method', ylabel=y_label, y_min=y_min, y_max=y_max, rotate= 90 ,palette="Set2", hue_order = ['Original', 'Shuffled'],
             figsize=(len(unique_models), 8), out_file_prefix=out_file_prefix)


def main():
    score_names = ['S_mean_mean', 'synergy_loewe_mean']
    split_types = ['leave_comb', 'leave_drug', 'leave_cell_line', 'random']

    metric = 'Pearsons'
    y_label = 'Pearson\'s Coefficient'
    # metric = 'test_RMSE'
    # y_label = 'RMSE'

    for score_name in score_names:
        result_dir = f'/home/grads/tasnina/Projects/SynVerse/outputs/k_0.05_{score_name}'
        # result_dir = f'/home/grads/tasnina/Projects/SynVerse/outputs/MARSY_data/k_0.05_{score_name}'
        score_name_str = score_name.split('_')[0]

        # result_dir = f'/home/grads/tasnina/Projects/SynVerse/outputs/sample_norm_0.99/k_0.05_{score_name}'

        for split_type in split_types:
            # plot for comparing models with each other. Also compare with one hot based model i.e., basleine
            result_file = f'output_{split_type}.tsv'
            result_file_path = os.path.join(result_dir, result_file)
            if not os.path.exists(result_file_path):
                print(f'file {result_file} does not exist. Continuing to next file.')
                continue
            result_df = pd.read_csv(result_file_path, sep='\t', index_col=None)
            #compute_RMSE from MSE
            for split in ['test', 'train', 'val']:
                result_df[f'{split}_RMSE'] = np.sqrt(result_df[f'{split}_MSE'])

            wrapper_plot_model_performance(copy.deepcopy(result_df),metric=metric, y_label=y_label, title=split_type, out_file_prefix = f'{result_dir}/{score_name_str}_{split_type}_{y_label}')
            wrapper_plot_compare_with_1hot(copy.deepcopy(result_df), metric=metric, y_label=y_label, title=split_type, out_file_prefix = f'{result_dir}/{score_name_str}_{split_type}_{y_label}')


            # plot for comparing models trained on original vs. shuffled features
            shuffled_result_file = f'output_{split_type}_shuffled.tsv'
            shuffled_result_file_path = os.path.join(result_dir, shuffled_result_file)
            if not os.path.exists(shuffled_result_file_path):
                print(f'file {shuffled_result_file_path} does not exist. Continuing to next file.')
                continue
            shuffled_result_df = pd.read_csv(shuffled_result_file_path, sep='\t', index_col=None)
            # compute_RMSE from MSE
            for split in ['test', 'train', 'val']:
                shuffled_result_df[f'{split}_RMSE'] = np.sqrt(shuffled_result_df[f'{split}_MSE'])
            wrapper_plot_compare_shuffled(result_df, shuffled_result_df, metric=metric, y_label=y_label,
                                         out_file_prefix=f'{result_dir}/{score_name_str}_{split_type}_{y_label}_shuffled')

            # plot for comparing models trained on original vs. rewired networks
            rewired_net_result_file = f'output_{split_type}_rewired.tsv'
            rewired_result_file_path = os.path.join(result_dir, rewired_net_result_file)
            if not os.path.exists(rewired_result_file_path):
                print(f'file {rewired_result_file_path} does not exist. Continuing to next file.')
                continue
            rewired_result_df = pd.read_csv(rewired_result_file_path, sep='\t', index_col=None)
            # compute_RMSE from MSE
            for split in ['test', 'train', 'val']:
                rewired_result_df[f'{split}_RMSE'] = np.sqrt(rewired_result_df[f'{split}_MSE'])

            wrapper_plot_compare_rewired(result_df, rewired_result_df, metric=metric, y_label=y_label,
                                         out_file_prefix=f'{result_dir}/{score_name_str}_{split_type}_{y_label}_rewired')



if __name__ == '__main__':
    main()