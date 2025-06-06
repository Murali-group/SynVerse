
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
from .plot_utils import *
from scipy.stats import mannwhitneyu, kruskal
from statsmodels.stats.multitest import multipletests
import matplotlib.colors as mcolors
import scipy.stats as stats
from itertools import combinations
import sys
import os

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ..parse_config import parse_config
import yaml
import argparse
from .parse_output_file import parse_output_files


custome_palette = [sns.color_palette('Paired')[i] for i in [2, 0, 6, 8, 10]] #green, blue, orange, purple
PALETTE =  custome_palette
original_model_color = custome_palette[0]
shuffle_palette = [custome_palette[i] for i in [0,2 ]]  #green. orange
randomized_palette=[custome_palette[i] for i in [0, 4 ]]
rewire_palette = [custome_palette[i] for i in [0, 3, 1 ]] #green, purple, blue

edge_color='#85929e'
bar_height=6
box_height=6
# edge_color = 'black'


def parse_args():
    parser = setup_opts()
    args = parser.parse_args()
    kwargs = vars(args)
    with open(args.config, 'r') as conf:
        # config_map = yaml.load(conf, Loader=yaml.FullLoader)
        config_map = yaml.safe_load(conf)
    return config_map, kwargs


def setup_opts():
    ## Parse command line args.
    parser = argparse.ArgumentParser(description="""Script to parse the file for training data and run the pipeline using them.""")
    # general parameters
    group = parser.add_argument_group('Main Options')
    group.add_argument('--config', type=str, default="code/config_files/archive/loewe_smiles_derived_feat.yaml", help="Configuration file for this script.")
    group.add_argument('--parse', action='store_true', help="Parse outfile.")
    group.add_argument('--plot', action='store_true', help="Generate plot.")

    return parser


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
        test_MSE_median=('test_MSE', 'median'),
        test_MSE_mean=('test_MSE', 'mean'),
        test_MSE_max=('test_MSE', 'max'),
        test_MSE_std=('test_MSE', 'std'),

        val_MSE_median=('val_MSE', 'median'),
        val_MSE_mean=('val_MSE', 'mean'),
        val_MSE_max=('val_MSE', 'max'),
        val_MSE_std=('val_MSE', 'std'),

        train_MSE_median=('train_MSE', 'median'),
        train_MSE_mean=('train_MSE', 'mean'),
        train_MSE_max=('train_MSE', 'max'),
        train_MSE_std=('train_MSE', 'std'),

        test_RMSE_median=('test_RMSE', 'median'),
        test_RMSE_mean=('test_RMSE', 'mean'),
        test_RMSE_max=('test_RMSE', 'max'),
        test_RMSE_std=('test_RMSE', 'std'),

        val_RMSE_median=('val_RMSE', 'median'),
        val_RMSE_mean=('val_RMSE', 'mean'),
        val_RMSE_max=('val_RMSE', 'max'),
        val_RMSE_std=('val_RMSE', 'std'),

        train_RMSE_median=('train_RMSE', 'median'),
        train_RMSE_mean=('train_RMSE', 'mean'),
        train_RMSE_max=('train_RMSE', 'max'),
        train_RMSE_std=('train_RMSE', 'std'),

        Pearsons_median=('Pearsons', 'median'),
        Pearsons_mean=('Pearsons', 'mean'),
        Pearsons_std=('Pearsons', 'std'),
        Pearsons_max=('Pearsons', 'max'),

        Spearman_median=('Spearman', 'median'),
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

        test_MSE_diff_median=('test_MSE_diff', 'median'),
        val_MSE_diff_median=('val_MSE_diff', 'median'),
        train_MSE_diff_median=('train_MSE_diff', 'median'),
        test_RMSE_diff_median=('test_RMSE_diff', 'median'),
        val_RMSE_diff_median=('val_RMSE_diff', 'median'),
        train_RMSE_diff_median=('train_RMSE_diff', 'median'),
        Pearsons_diff_median=('Pearsons_diff', 'median'),
        Spearman_diff_median=('Spearman_diff', 'median'),

    ).reset_index()
    return aggregated_results


def aggregate_scores(df):
    aggregated_results = df.groupby(['drug_features', 'cell_features', 'feature_filter','Model']).agg(
        test_MSE_median=('test_MSE', 'median'),
        test_MSE_mean=('test_MSE', 'mean'),
        test_MSE_max=('test_MSE', 'max'),
        test_MSE_std=('test_MSE', 'std'),

        val_MSE_median=('val_MSE', 'median'),
        val_MSE_mean=('val_MSE', 'mean'),
        val_MSE_max=('val_MSE', 'max'),
        val_MSE_std=('val_MSE', 'std'),

        train_MSE_median=('train_MSE', 'median'),
        train_MSE_mean=('train_MSE', 'mean'),
        train_MSE_max=('train_MSE', 'max'),
        train_MSE_std=('train_MSE', 'std'),

        test_RMSE_median=('test_RMSE', 'median'),
        test_RMSE_mean=('test_RMSE', 'mean'),
        test_RMSE_max=('test_RMSE', 'max'),
        test_RMSE_std=('test_RMSE', 'std'),

        val_RMSE_median=('val_RMSE', 'median'),
        val_RMSE_mean=('val_RMSE', 'mean'),
        val_RMSE_max=('val_RMSE', 'max'),
        val_RMSE_std=('val_RMSE', 'std'),

        train_RMSE_median=('train_RMSE', 'median'),
        train_RMSE_mean=('train_RMSE', 'mean'),
        train_RMSE_max=('train_RMSE', 'max'),
        train_RMSE_std=('train_RMSE', 'std'),

        Pearsons_median=('Pearsons', 'median'),
        Pearsons_mean=('Pearsons', 'mean'),
        Pearsons_std=('Pearsons', 'std'),
        Pearsons_max=('Pearsons', 'max'),

        Spearman_median=('Spearman', 'median'),
        Spearman_mean=('Spearman', 'mean'),
        Spearman_std=('Spearman', 'std'),
        Spearman_max=('Spearman', 'max'),

    ).reset_index()

    # Restore categorical ordering for 'Model'
    aggregated_results['Model'] = pd.Categorical(aggregated_results['Model'], categories=model_name_mapping.values(), ordered=True)

    # Explicitly sort again based on categorical order
    aggregated_results = aggregated_results.sort_values('Model')
    aggregated_results['Model'] = aggregated_results['Model'].astype(str)
    return aggregated_results

def compute_average_and_significance(df, measure, alt='greater'):

    df = compute_difference_with_1hot(df)
    # Group by and compute aggregation metrics
    aggregated_results = compute_average_with_1hot_diff(df)

    # Define a function to compute the Mann-Whitney U test for a group
    def compute_mannwhitney(group):
        stat, p_value = mannwhitneyu(group[measure], group[f'{measure}_baseline'], alternative=alt)
        return pd.Series({'stat': stat, 'p_value': p_value})


    # Compute significance of the test_loss compared to test_loss_baseline
    significance_results = (
        df.groupby(['drug_features', 'cell_features', 'feature_filter','Model'])
        .apply(compute_mannwhitney)
        .reset_index()
    )

    # Apply multiple testing correction
    significance_results.dropna(inplace=True)
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



def wrapper_plot_model_performance_subplots(df, ft_filt_wise_1hot, metric, y_label, title, orientation='vertical', out_file_prefix=None):

    # remove one-hot based model
    df = df[df['Model'] != 'One hot']

    if (metric == 'Pearsons') | (metric == 'Spearman'):
        y_max = 1
        y_min = min(min(df[metric]), 0.7)
    else:
        y_max = None
        y_min = None
    # Get unique feature filters
    feature_filters = df['feature_filter'].unique()

    if orientation =='vertical':
        figsize = (2.5 * len(feature_filters), box_height)
        rotation = 90
    elif orientation=='horizontal':
        figsize = (box_height, 2.5 * len(feature_filters))
        rotation = 0

    # #without baseline
    # box_plot_subplots(
    #     df, x='Model', y=metric,
    #     ylabel=y_label,
    #     hue=None, hue_order=None,
    #     feature_filters=feature_filters, rotate=rotation, y_min=y_min, y_max=y_max,
    #     figsize= figsize,
    #     color=original_model_color,
    #     width=0.7,
    #     dodge=True, edgecolor=edge_color, legend=False, bg_colors = ["#A9A9A9", "white" ],orientation=orientation,
    #     out_file_prefix=f'{out_file_prefix}',
    # )

    # with baseline
    box_plot_subplots(
        df, x='Model', y=metric,
        ylabel=y_label,
        hue=None, hue_order=None,
        feature_filters=feature_filters, rotate=rotation, y_min=y_min, y_max=y_max,
        figsize=figsize,
        color=original_model_color,
        width=0.7, ft_filt_wise_1hot=ft_filt_wise_1hot,
        dodge=True, edgecolor=edge_color, legend=False, bg_colors = ["#A9A9A9", "white" ], orientation=orientation,
        out_file_prefix=f'{out_file_prefix}_baseline',
    )

    print(title)


def wrapper_plot_compare_with_1hot_subplots(df, metric, y_label, title, orientation='vertical', out_file_prefix=None):
    df_1hot_diff = compute_difference_with_1hot(df)
    if df_1hot_diff.empty:
        return

    # df_1hot_diff.to_csv(f'{out_file_prefix}_with_baseline.tsv', sep='\t')

    # Pair plot for comparing each model with baseline across individual runs
    # pair_plot(df_1hot_diff, metric, out_file_prefix=f'{out_file_prefix}')

    if metric == 'Pearsons':
        y_min = min(-0.1, min(df_1hot_diff[f'{metric}_diff']))
        y_max = max(0.1, max(df_1hot_diff[f'{metric}_diff']))
    else:
        y_min = None
        y_max = None

    # Remove one-hot based model
    df_1hot_diff = df_1hot_diff[df_1hot_diff['Model'] != 'One hot']

    # Get unique feature filters
    feature_filters = df_1hot_diff['feature_filter'].unique()



    if orientation =='vertical':
        figsize = (2.5 * len(feature_filters), box_height)
        rotation = 90
    elif orientation=='horizontal':
        figsize = (box_height, 2.5 * len(feature_filters))
        rotation = 0

    # Call box plot with subplots
    box_plot_subplots(
        df_1hot_diff, x='Model', y=f'{metric}_diff',
        ylabel='Improvement over baseline (Pearson\'s)',
        feature_filters=feature_filters, rotate=rotation, y_min=y_min, y_max=y_max,
        figsize=figsize,
        color=original_model_color,
        width=0.7, dodge=False, edgecolor='black',zero_line=True, legend=False, orientation=orientation,
        out_file_prefix=f'{out_file_prefix}'
    )


def wrapper_plot_compare_rewired_subplots(result_df, rewired_result_df,ft_filt_wise_1hot, metric, y_label, orientation='vertical', out_file_prefix=None):

        df = pd.concat([result_df, rewired_result_df], axis=0)
        # df = set_model_names(df)


        #keeps models  which I ran on rewired network
        rewired_model_names = df[df['rewired']==True]['Model'].unique()
        df = df[df['Model'].isin(rewired_model_names)]

        if (metric == 'Pearsons') | (metric == 'Spearman'):
            y_max=1
            y_min = min(min(df[metric]), 0)
        else:
            y_max=None
            y_min=None
        # Get unique feature filters
        feature_filters = df['feature_filter'].unique()

        if orientation == 'vertical':
            figsize = (2.5 * len(feature_filters), box_height)
            rotation = 90
        elif orientation == 'horizontal':
            figsize = (box_height, 2.5 * len(feature_filters))
            rotation = 0

        # box plot with subplots with baseline
        box_plot_subplots(
            df, x='Model', y=metric,
            ylabel=y_label,
            hue='rewire_method', hue_order = ['Original', 'SA', 'SM'],
            feature_filters=feature_filters, rotate=rotation, y_min=y_min, y_max=y_max,
            figsize=figsize,
            ft_filt_wise_1hot=ft_filt_wise_1hot,
            # palette="Set2",
            palette=rewire_palette,
            width=0.9, dodge=True, edgecolor='black', bg_colors = ["#A9A9A9", "white" ], orientation=orientation,
            out_file_prefix=f'{out_file_prefix}'
        )


def wrapper_plot_compare_shuffled_subplots(result_df, shuffled_result_df, ft_filt_wise_1hot, metric, y_label, out_file_prefix=None, orientation='vertical'):
    df = pd.concat([result_df, shuffled_result_df], axis=0)
    # df = set_model_names(df)

    # keeps models  which I ran with shuffled features
    shuffled_model_names = df[df['shuffled'] == True]['Model'].unique()
    df = df[df['Model'].isin(shuffled_model_names)]

    # modify model name to look good on plot
    # df['Model'] = df['Model'].str.replace(r'\(', r'\n(', regex=True)

    if (metric == 'Pearsons') | (metric == 'Spearman'):
        y_max = 1
        y_min = min(min(df[metric]), 0.7)
    else:
        y_max = None
        y_min = None
    # modify model name to look good on plot
    feature_filters = df['feature_filter'].unique()


    if orientation =='vertical':
        figsize = (2.5 * len(feature_filters), box_height)
        rotation = 90
    elif orientation=='horizontal':
        figsize = (box_height, 2.5 * len(feature_filters))
        rotation = 0
    # box plot with subplots with baseline
    box_plot_subplots(
        df, x='Model', y=metric,
        ylabel=y_label,
        hue='shuffle_method', hue_order = ['Original', 'Shuffled'],
        feature_filters=feature_filters, rotate=rotation, y_min=y_min, y_max=y_max,
        figsize=figsize,
        ft_filt_wise_1hot=ft_filt_wise_1hot,
        palette=shuffle_palette,
        width=0.8, dodge=True, edgecolor='black', bg_colors = ["#A9A9A9", "white" ], orientation=orientation,
        out_file_prefix=f'{out_file_prefix}'
    )

def wrapper_plot_compare_randomized_subplots(result_df, randomized_result_df, ft_filt_wise_1hot, metric, y_label, out_file_prefix=None, orientation='vertical'):
    df = pd.concat([result_df, randomized_result_df], axis=0)
    # df = set_model_names(df)

    # keeps models  which I ran with randomized_score
    randomized_model_names = df[df['randomized_score'] == True]['Model'].unique()
    df = df[df['Model'].isin(randomized_model_names)]

    print(len(df), df.head(5))
    if (metric == 'Pearsons') | (metric == 'Spearman'):
        y_max = 1
        y_min = min(min(df[metric]), 0.7)
    else:
        y_max = None
        y_min = None
    # modify model name to look good on plot
    feature_filters = df['feature_filter'].unique()


    if orientation =='vertical':
        figsize = (2.5 * len(feature_filters), box_height)
        rotation = 90
    elif orientation=='horizontal':
        figsize = (box_height, 2.5 * len(feature_filters))
        rotation = 0
    # box plot with subplots with baseline
    box_plot_subplots(
        df, x='Model', y=metric,
        ylabel=y_label,
        hue='randomized_method', hue_order = ['Original', 'Randomized'],
        feature_filters=feature_filters, rotate=rotation, y_min=y_min, y_max=y_max,
        figsize=figsize,
        ft_filt_wise_1hot=ft_filt_wise_1hot,
        palette=randomized_palette,
        width=0.8, dodge=True, edgecolor='black', bg_colors = ["#A9A9A9", "white" ], orientation=orientation,
        out_file_prefix=f'{out_file_prefix}'
    )


def pairwise_significance_test_wrapper(df, group_by_cols, compare_based_on,
                                       measure, alt='two-sided', test="mannwhitney", out_file_prefix=None):
    """
    Performs pairwise statistical tests within groups defined by `group_by_cols`.

    Parameters:
    df (pd.DataFrame): Input DataFrame.
    group_by_cols (list): Columns used to group data before comparison.
    compare_based_on (str): Column used for pairwise comparisons within each group.
    test (str): Statistical test to use. Options: "mannwhitney", "ttest", "kruskal".

    Returns:
    pd.DataFrame: A DataFrame with p-values for each pairwise comparison.
    """
    results = []

    # Group by the specified columns
    grouped = df.groupby(group_by_cols)

    for group_key, group_df in grouped:
        unique_categories = group_df[compare_based_on].unique()

        # Get all possible pairwise comparisons
        for cat1, cat2 in combinations(unique_categories, 2):
            data1 = group_df[group_df[compare_based_on] == cat1][measure]
            data2 = group_df[group_df[compare_based_on] == cat2][measure]

            data1.dropna(inplace=True) #if a few runs were not complete.
            data2.dropna(inplace=True) #if a few runs were not complete.
            # Choose test
            if test == "mannwhitney":
                p_value = stats.mannwhitneyu(data1, data2, alternative=alt).pvalue
            elif test == "ttest":
                p_value = stats.ttest_ind(data1, data2, equal_var=False).pvalue
            elif test == "kruskal":
                p_value = stats.kruskal(data1, data2).pvalue
            else:
                raise ValueError("Unsupported test. Choose from 'mannwhitney', 'ttest', or 'kruskal'.")

            # Store results
            results.append({
                "Group": group_key,
                "Comparison": f"{cat1} vs {cat2}",
                "raw_p_value": p_value
            })

    sig_df = pd.DataFrame(results)
    # Apply Benjamini-Hochberg correction if necessary
    if not sig_df.empty:
        sig_df["adjusted_p_value"] = multipletests(sig_df["raw_p_value"], method="fdr_bh")[1]

    if out_file_prefix:
        sig_df.to_csv(f"{out_file_prefix}.tsv", sep='\t')

    # Convert results to DataFrame
    return sig_df


def compare_across_models(df, divide_by, group_by_cols, metric, sorted_models = None):
    """
    For each unique value in `divide_by`:
      1) Run a Kruskal–Wallis omnibus test across the groups defined by `group_by_cols`.
      2) Run all pairwise Mann–Whitney U tests between those groups.
      3) Apply Benjamini–Hochberg FDR correction to the pairwise p-values within each subset.

    Returns
    -------
    omnibus_df : pandas.DataFrame
        Columns: [divide_by, 'H_statistic', 'p_value']
    pairwise_df : pandas.DataFrame
        Columns: [divide_by, 'group1', 'group2', 'p_value', 'p_adj', 'reject']
    """

    # sort df according to their performance.
    # if sorted_models:
    #     # remove multiple presence of 'One hot' from sorted_model list
    #     sorted_models = [x for x in sorted_models if x != 'One hot']
    #     sorted_models.append('One hot')
    #     df['Model'] = pd.Categorical(df['Model'], categories=sorted_models, ordered=True)
    #     df = df.sort_values('Model')
    #     df['Model'] = df['Model'].astype(str)

    #remove one hot model
    df = df[df['Model'] != 'One hot']

    omnibus_results = []
    pairwise_results = []

    for val in df[divide_by].unique():
        subset = df[df[divide_by] == val]
        grouped = subset.groupby(group_by_cols)

        # omnibus
        samples = [grp[metric].values for _, grp in grouped]
        if len(samples) >= 2:
            H_stat, p_kw = kruskal(*samples, nan_policy='omit')
        else:
            H_stat, p_kw = np.nan, np.nan

        omnibus_results.append({
            divide_by: val,
            'H_statistic': H_stat,
            'p_value': p_kw
        })

        # pairwise
        items = list(grouped)  # list of (name, DataFrame)
        if sorted_models:
            items = sorted(items, key=lambda x: sorted_models.index(x[0]) if x[0] in sorted_models else float('inf'))

        for (name1, grp1), (name2, grp2) in combinations(items, 2):

            try:
                _, p_pair_two_sided = mannwhitneyu(grp1[metric], grp2[metric], alternative='two-sided')
                _, p_pair_less = mannwhitneyu(grp1[metric], grp2[metric], alternative='less')
                _, p_pair_greater = mannwhitneyu(grp1[metric], grp2[metric], alternative='greater')

            except ValueError:
                print('error occured during significance test')

            g1 = "_".join(map(str, name1 if isinstance(name1, tuple) else [name1]))
            g2 = "_".join(map(str, name2 if isinstance(name2, tuple) else [name2]))
            pairwise_results.append({
                divide_by: val,
                'group1': g1,
                'group2': g2,
                'p_value_two_sided': p_pair_two_sided,
                'p_value_less': p_pair_less,
                'p_value_greater': p_pair_greater

            })

    omnibus_df = pd.DataFrame(omnibus_results)
    pairwise_df = pd.DataFrame(pairwise_results)

    # apply BH correction **within each** divide_by family
    def adjust(group, pval_cols=None, method='fdr_bh'):
        if pval_cols is None:
            pval_cols = ['p_value_two_sided', 'p_value_less', 'p_value_greater']

        for pval_col in pval_cols:
            pvals = group[pval_col].fillna(1.0)
            reject, p_adj, _, _ = multipletests(pvals, method=method)

            group = group.assign(**{
                f'adj_{pval_col}': p_adj,
                f'reject_{pval_col}': reject
            })

        return group

    # pairwise_df = pairwise_df.groupby(divide_by, group_keys=False).apply(adjust)
    pairwise_df = pairwise_df.groupby(divide_by, group_keys=False).apply(adjust)

    return omnibus_df, pairwise_df



def compute_performance_retained_wrapper(df, group_by_cols, compare_based_on,
                             measure, out_file_prefix=None):
    """
    Performs pairwise statistical tests within groups defined by `group_by_cols`.

    Parameters:
    df (pd.DataFrame): Input DataFrame.
    group_by_cols (list): Columns used to group data before comparison.
    compare_based_on (str): Column used for pairwise comparisons within each group.
    test (str): Statistical test to use. Options: "mannwhitney", "ttest", "kruskal".

    Returns:
    pd.DataFrame: A DataFrame with p-values for each pairwise comparison.
    """
    results = []

    # Group by the specified columns
    grouped = df.groupby(group_by_cols)

    for group_key, group_df in grouped:
        unique_categories = group_df[compare_based_on].unique()

        # Get all possible pairwise comparisons
        for cat1, cat2 in combinations(unique_categories, 2):
            data1 = group_df[group_df[compare_based_on] == cat1][measure]
            data2 = group_df[group_df[compare_based_on] == cat2][measure]

            # data1.dropna(inplace=True) #if a few runs were not complete.
            # data2.dropna(inplace=True) #if a few runs were not complete.

            median_1 = data1.median()
            median_2 = data2.median()

            retained  = min(median_1, median_2)/max(median_1, median_2)
            outperformed = max(median_1, median_2)/min(median_1, median_2)

            # Store results
            results.append({
                "Group": group_key,
                "Comparison": f"{cat1} vs {cat2}",
                "performance_retained": retained,
                "outperformed": outperformed
            })

    performance_comp_df = pd.DataFrame(results)
    # Apply Benjamini-Hochberg correction if necessary

    if out_file_prefix:
        performance_comp_df.to_csv(f"{out_file_prefix}.tsv", sep='\t')

    # Convert results to DataFrame
    return performance_comp_df
def generate_plots(params):
    score_name_mapping = {'S_mean_mean':'S', 'synergy_loewe_mean':'Loewe'}
    score_name = params.score_name
    split_types = [split['type'] for split in params.splits]

    plot_metrics = [{'metric': 'Pearsons', 'y_label':'PCC', 'alt': 'greater' },
                    {'metric': 'test_RMSE', 'y_label':'RMSE', 'alt': 'less' }]
    orientations= [ 'vertical']

    for metric_info in plot_metrics:
        metric = metric_info['metric']
        y_label = metric_info['y_label']
        alt= metric_info['alt'] #alternate hypothesis for significance test

        for orientation in orientations:
            result_dir = f'/{params.out_dir}/k_{params.abundance}_{score_name}'
            score_name_str = score_name_mapping[score_name]

            for split_type in split_types:
                # plot for comparing models with each other. Also compare with one hot based model i.e., basleine
                result_file = f'output_{split_type}.tsv'
                result_file_path = os.path.join(result_dir, result_file)
                if not os.path.exists(result_file_path):
                    print(f'file {result_file} does not exist. Continuing to next file.')
                    continue
                result_df = pd.read_csv(result_file_path, sep='\t', index_col=None)
                # if the model_name mapping is not available, we do not want to plot the model's performance.
                result_df.dropna(subset=['Model'], inplace=True)

                #compute_RMSE from MSE
                for split in ['test', 'train', 'val']:
                    result_df[f'{split}_RMSE'] = np.sqrt(result_df[f'{split}_MSE'])

                # compute average, median, min, max of scores across runs.
                df_avg = aggregate_scores(result_df)
                df_avg.to_csv(f'{result_dir}/{score_name_str}_{split_type}_aggreagred.tsv', sep='\t')

                significance_df = compute_average_and_significance(copy.deepcopy(result_df), metric, alt=alt)
                significance_df.to_csv(
                    f'{result_dir}/significance_baseline_diff_{score_name_str}_{split_type}_{metric}.tsv', sep='\t')


                #Kruskal test
                sorted_models = df_avg.sort_values(by=f'{metric}_median', ascending=False if metric=='Pearsons' else True)['Model'].tolist() #return model names ordered accoring to their performance
                kruskal_df, pair_wise_sig_df = compare_across_models(copy.deepcopy(result_df), divide_by ='feature_filter', group_by_cols='Model', metric=metric, sorted_models=sorted_models)
                kruskal_df.to_csv(f'{result_dir}/{score_name_str}_{split_type}_{metric}_kruskal.tsv', sep='\t')
                pair_wise_sig_df.to_csv(f'{result_dir}/{score_name_str}_{split_type}_{metric}_pairwise_model_significance.tsv', sep='\t')

                # get feature_filter wise one_hot model's  performance
                df_1hot = df_avg[df_avg['Model'] == 'One hot']
                ft_filt_wise_1hot = dict(zip(df_1hot['feature_filter'], df_1hot[f'{metric}_median']))

                wrapper_plot_model_performance_subplots(copy.deepcopy(result_df),ft_filt_wise_1hot, metric=metric, y_label=y_label, title=split_type, orientation=orientation, out_file_prefix =f'{result_dir}/plot/{orientation}/{score_name_str}_{split_type}_{metric}')


                # plot for comparing models trained on original vs. shuffled features
                shuffled_result_file = f'output_{split_type}_shuffled.tsv'
                shuffled_result_file_path = os.path.join(result_dir, shuffled_result_file)
                if not os.path.exists(shuffled_result_file_path):
                    print(f'file {shuffled_result_file_path} does not exist. Continuing to next file.')
                    continue
                shuffled_result_df = pd.read_csv(shuffled_result_file_path, sep='\t', index_col=None)
                # if the model_name mapping is not available, we do not want to plot the model's performance.
                shuffled_result_df.dropna(subset=['Model'], inplace=True)

                # compute_RMSE from MSE
                for split in ['test', 'train', 'val']:
                    shuffled_result_df[f'{split}_RMSE'] = np.sqrt(shuffled_result_df[f'{split}_MSE'])

                #save aggregated file
                shuffled_result_df_agg = shuffled_result_df.groupby(['Model', 'shuffle_method']).agg(
                    test_RMSE_mean=('test_RMSE', 'mean'),
                    test_RMSE_median=('test_RMSE', 'median'),
                    Pearsons_mean=('Pearsons', 'mean'),
                    Pearsons_median=('Pearsons', 'median'),
                    Spearman_mean=('Spearman', 'mean'),
                    Spearman_median=('Spearman', 'median')
                )
                shuffled_result_df_agg.to_csv(f'{result_dir}/shuffled_{score_name_str}_{split_type}_{metric}_aggregated.tsv', sep='\t')

                pairwise_significance_test_wrapper(pd.concat([result_df, shuffled_result_df], axis=0), group_by_cols=['Model', 'feature_filter'],
                                                   compare_based_on='shuffle_method', measure=metric, out_file_prefix=f'{result_dir}/significance_shuffled_{score_name_str}_{split_type}_{metric}')
                wrapper_plot_compare_shuffled_subplots(result_df, shuffled_result_df, ft_filt_wise_1hot, metric=metric, y_label=y_label, orientation=orientation,
                                             out_file_prefix=f'{result_dir}/plot/{orientation}/shuffled_{score_name_str}_{split_type}_{metric}')

                #*********************

                print('\n\n starting randomized plot')

                # plot for comparing models trained on original vs. shuffled features
                random_score_result_file = f'output_{split_type}_randomized.tsv'
                random_score_result_file_path = os.path.join(result_dir, random_score_result_file)
                if not os.path.exists(random_score_result_file_path):
                    print(f'file {random_score_result_file_path} does not exist. Continuing to next file.')
                    continue
                random_score_result_df = pd.read_csv(random_score_result_file_path, sep='\t', index_col=None)
                # if the model_name mapping is not available, we do not want to plot the model's performance.
                random_score_result_df.dropna(subset=['Model'], inplace=True)

                # compute_RMSE from MSE
                for split in ['test', 'train', 'val']:
                    random_score_result_df[f'{split}_RMSE'] = np.sqrt(random_score_result_df[f'{split}_MSE'])

                # save aggregated file
                random_score_result_df_agg = random_score_result_df.groupby(['Model', 'randomized_method']).agg(
                    test_RMSE_mean=('test_RMSE', 'mean'),
                    test_RMSE_median=('test_RMSE', 'median'),
                    Pearsons_mean=('Pearsons', 'mean'),
                    Pearsons_median=('Pearsons', 'median'),
                    Spearman_mean=('Spearman', 'mean'),
                    Spearman_median=('Spearman', 'median')
                )
                random_score_result_df_agg.to_csv(
                    f'{result_dir}/randomized_score_{score_name_str}_{split_type}_{metric}_aggregated.tsv', sep='\t')

                pairwise_significance_test_wrapper(pd.concat([result_df, random_score_result_df], axis=0),
                                                   group_by_cols=['Model', 'feature_filter'],
                                                   compare_based_on='randomized_method', measure=metric,
                                                   out_file_prefix=f'{result_dir}/significance_randomized_score_{score_name_str}_{split_type}_{metric}')
                wrapper_plot_compare_randomized_subplots(result_df, random_score_result_df, ft_filt_wise_1hot, metric=metric,
                                                       y_label=y_label, orientation=orientation,
                                                       out_file_prefix=f'{result_dir}/plot/{orientation}/randomized_score_{score_name_str}_{split_type}_{metric}')
                if metric=='Pearsons': #computed retained performance by rewired models
                    compute_performance_retained_wrapper(pd.concat([result_df, random_score_result_df], axis=0),
                                          group_by_cols=['Model'],
                                          compare_based_on='randomized_method', measure=metric,
                                          out_file_prefix=f'{result_dir}/retained_randomized_score_{score_name_str}_{split_type}_{metric}')
                #*********************


                # plot for comparing models trained on original vs. rewired networks
                rewired_net_result_file = f'output_{split_type}_rewired.tsv'
                rewired_result_file_path = os.path.join(result_dir, rewired_net_result_file)
                if not os.path.exists(rewired_result_file_path):
                    print(f'file {rewired_result_file_path} does not exist. Continuing to next file.')
                    continue
                rewired_result_df = pd.read_csv(rewired_result_file_path, sep='\t', index_col=None)
                # if the model_name mapping is not available, we do not want to plot the model's performance.
                rewired_result_df.dropna(subset=['Model'], inplace=True)

                # compute_RMSE from MSE
                for split in ['test', 'train', 'val']:
                    rewired_result_df[f'{split}_RMSE'] = np.sqrt(rewired_result_df[f'{split}_MSE'])
                #save aggregated results
                rewired_result_df_agg = rewired_result_df.groupby(['Model', 'rewire_method']).agg(
                    test_RMSE_mean=('test_RMSE', 'mean'),
                    test_RMSE_median=('test_RMSE', 'median'),
                    Pearsons_mean=('Pearsons', 'mean'),
                    Pearsons_median=('Pearsons', 'median'),
                    Spearman_mean=('Spearman', 'mean'),
                    Spearman_median=('Spearman', 'median')
                )
                rewired_result_df_agg.to_csv(f'{result_dir}/rewired_{score_name_str}_{split_type}_{metric}_aggregated.tsv', sep='\t')

                pairwise_significance_test_wrapper(pd.concat([result_df, rewired_result_df], axis=0),
                                                   group_by_cols=['Model', 'feature_filter'],
                                                   compare_based_on='rewire_method', measure=metric, alt = alt,
                                                   out_file_prefix=f'{result_dir}/significance_rewired_{score_name_str}_{split_type}_{metric}')

                if metric=='Pearsons': #computed retained performance by rewired models
                    compute_performance_retained_wrapper(pd.concat([result_df, rewired_result_df], axis=0),
                                          group_by_cols=['Model'],
                                          compare_based_on='rewire_method', measure=metric,
                                          out_file_prefix=f'{result_dir}/retained_rewired_{score_name_str}_{split_type}_{metric}')
                wrapper_plot_compare_rewired_subplots(result_df, rewired_result_df, ft_filt_wise_1hot, metric=metric, y_label=y_label, orientation=orientation,
                                             out_file_prefix=f'{result_dir}/plot/{orientation}/rewired_{score_name_str}_{split_type}_{metric}')

                print(f'done {split_type}')


def main(params, **kwargs):
    result_dir = f'/{params.out_dir}/k_{params.abundance}_{params.score_name}/'
    if kwargs.get('parse'):
        parse_output_files(result_dir)
    if kwargs.get('plot'):
        generate_plots(params)

if __name__ == '__main__':
    config_map, kwargs = parse_args()
    # parse input_files and params from config_file. If some params are overlapping across config_map and kwargs, then prioritize kwargs.
    _, params = parse_config(config_map, **kwargs)
    main(params, **kwargs)
