import copy
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew


feature_filters = ['D_ECFP_4_MACCS_MFP_d1hot_mol_graph_smiles_C_c1hot',
                  'D_d1hot_target_C_c1hot', 'D_d1hot_C_c1hot_genex_genex_lincs_1000']

#rename the long model names containing features, preprocessing and encoder name to suitable name for plot.
model_name_mapping = {'d1hot + c1hot': 'One hot','d1hot_std_comp_True + c1hot_std_comp_True': 'One hot (AE)',
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

def set_model_names(df):
    # Create x-tick labels combining 'drug_features' and 'cell_features'
    df['Model'] = df['drug_features'] + " + " + df['cell_features']
    df['Model'] = df['Model'].astype(str).apply(lambda x: model_name_mapping.get(x, x))
    # Set categorical dtype with the correct order
    df['Model'] = pd.Categorical(df['Model'], categories=model_name_mapping.values(), ordered=True)

    # Explicitly sort the DataFrame based on categorical order
    df = df.sort_values('Model')
    return df


def bar_plot_subplots(data, x, y, ylabel, feature_filters, hue=None, y_min=None, y_max=None, rotate=0, palette=None,
    color=None, hue_order=None, out_file_prefix=None, figsize=(6, 8), width=0.5, title='', dodge=True, ft_filt_wise_1hot=None, legend='auto',
                      edgecolor = 'black'):
    """
    Creates subplots for each `feature_filter`, with each subplot containing grouped bar plots.
    """

    # Create subplots dynamically based on the number of `feature_filters`

    row_counts = [len(data[data['feature_filter'] == feature_filter]) for feature_filter in feature_filters]
    total_rows = sum(row_counts)
    widths = [row_count / total_rows for row_count in row_counts]

    # Create subplots with dynamic widths
    fig, axes = plt.subplots(1, len(feature_filters),
                             figsize=figsize,
                             gridspec_kw={'width_ratios': widths}, sharey=True)

    if len(feature_filters) == 1:  # Ensure `axes` is iterable
        axes = [axes]

    handles, labels = None, None

    for ax, feature_filter in zip(axes, feature_filters):
        subset = data[data['feature_filter'] == feature_filter].reset_index(drop=True)
        if subset.empty:
            continue

        if hue is not None:
            # Plot grouped bar plot in subplot
            bar = sns.barplot(
                data=subset, x=x, y=y, hue=hue, hue_order=hue_order,
                dodge=dodge, width=width, palette=palette, ax=ax, errorbar="sd",
                edgecolor=edgecolor, linewidth=0.6,
            )
        else:
            bar = sns.barplot(
                data=subset, x=x, y=y,
                dodge=dodge, width=width, color=color, ax=ax, errorbar="sd",
                edgecolor=edgecolor, linewidth=0.6, gap=0.05
            )
        # Formatting
        ax.set_xlabel('')
        ax.set_ylabel(ylabel, fontsize=14)

        # Get the unique model names
        models = list(subset['Model'].unique())
        # Set the tick positions corresponding to each model label
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=rotate, fontsize=12)

        ax.yaxis.grid(True, linestyle='--', color='grey', alpha=0.6, linewidth=0.6)

        # Set y-axis limits if provided
        if (y_min is not None) and (y_max is not None):
            ax.set_ylim(y_min, y_max)

        if ft_filt_wise_1hot:
            ax.axhline(y=ft_filt_wise_1hot[feature_filter], color='red', linestyle='--', linewidth=0.8)

        # Store legend handles and labels for a shared legend
        if legend and bar.get_legend() is not None:
            handles, labels = bar.get_legend_handles_labels()
            ax.get_legend().remove()

    # Common y-label for all subplots
    if legend and handles:
        fig.legend(handles, labels, loc='upper center', ncol=len(data[hue].unique()), frameon=False, title=None,
                   fontsize=12,
                   bbox_to_anchor=(0.5, 1.02))

    plt.tight_layout()
    if out_file_prefix is not None:
        os.makedirs(os.path.dirname(out_file_prefix), exist_ok=True)
        plot_file = f"{out_file_prefix}_barplot.pdf"
        plt.savefig(plot_file, bbox_inches='tight')
        print(f'Saved file: {plot_file}')

    plt.show()

# def box_plot_subplots(data, x, y, ylabel, feature_filters, hue=None, y_min=None, y_max=None, rotate=0, palette=None,
#                       color=None, hue_order=None, out_file_prefix=None, figsize=(6, 8), width=0.5, title='',
#                       dodge=True, zero_line=False, legend='auto', edgecolor='black', ft_filt_wise_1hot=None, bg_colors=None):
#     """
#     Creates subplots for each `feature_filter`, with each subplot containing box plots.
#     """
#
#     # Create subplots dynamically based on the number of `feature_filters`
#     # Create subplots with dynamic widths based on the number of rows in each subset
#     row_counts = [len(data[data['feature_filter'] == feature_filter]) for feature_filter in feature_filters]
#     total_rows = sum(row_counts)
#     widths = [row_count / total_rows for row_count in row_counts]
#
#     # Create subplots with dynamic widths
#     fig, axes = plt.subplots(1, len(feature_filters),
#                              figsize=figsize,
#                              gridspec_kw={'width_ratios': widths}, sharey=True)
#
#     # fig, axes = plt.subplots(1, len(feature_filters), figsize=figsize, sharey=True)
#     if len(feature_filters) == 1:  # Ensure `axes` is iterable
#         axes = [axes]
#
#     for ax, feature_filter in zip(axes, feature_filters):
#         subset = data[data['feature_filter'] == feature_filter].reset_index(drop=True)
#         if subset.empty:
#             continue
#
#         if bg_colors:
#             # Determine unique models for the x-axis and add alternating background colors.
#             models = list(subset['Model'].unique())
#             ax.set_axisbelow(True)  # Ensure the background spans remain behind other plot elements.
#             for i, model in enumerate(models):
#                 bg_color = bg_colors[0] if i % 2 == 0 else bg_colors[1]
#                 # Draw a background rectangle spanning from i-0.5 to i+0.5 on the x-axis.
#                 ax.axvspan(i - 0.5, i + 0.5, facecolor=bg_color, alpha=0.3, zorder=0)
#
#         # Plot box plot in subplot
#         if color:
#             box = sns.boxplot(
#                 data=subset, x=x, y=y, dodge=dodge, width=width, ax=ax, linewidth=0.8,
#                 color = color, boxprops={'edgecolor':edgecolor}
#             )
#         else:
#             box = sns.boxplot(data=subset, x=x, y=y, hue=hue, hue_order=hue_order, dodge=dodge, width=width, palette=palette, ax=ax,
#                         linewidth=0.8, legend=legend, boxprops={'edgecolor':edgecolor})
#
#         # Formatting
#         # ax.set_title(f"{feature_filter}", fontsize=14)
#         ax.set_xlabel('')
#         ax.set_ylabel(ylabel, fontsize=14)
#
#         # Get the unique model names
#         # models = list(subset['Model'].unique())
#         # Set the tick positions corresponding to each model label
#         ax.set_xticks(range(len(models)))
#         ax.set_xticklabels(models, rotation=rotate, fontsize=12)
#
#         ax.yaxis.grid(True, linestyle='--', color='grey', alpha=0.6, linewidth=0.6)
#
#         # Set y-axis limits if provided
#         if (y_min is not None) and (y_max is not None):
#             ax.set_ylim(y_min, y_max)
#
#         if ft_filt_wise_1hot:
#             ax.axhline(y=ft_filt_wise_1hot[feature_filter], color='red', linestyle='--', linewidth=0.8)
#
#         # Add zero line if enabled
#         if zero_line:
#             ax.axhline(y=0, color='red', linestyle='--', linewidth=0.8)
#
#         # Store legend handles and labels for a shared legend
#         if legend and box.get_legend() is not None:
#             handles, labels = box.get_legend_handles_labels()
#             ax.get_legend().remove()
#     # Common y-label for all subplots
#     # fig.text(-0.01, 0.5, ylabel, va='center', rotation='vertical', fontsize=14)
#     if legend and handles:
#         fig.legend(handles, labels, loc='upper center', ncol=len(data[hue].unique()), frameon=False, title=None, fontsize=12,
#                    bbox_to_anchor=(0.5, 1.02))
#
#     plt.tight_layout()
#     if out_file_prefix is not None:
#         os.makedirs(os.path.dirname(out_file_prefix), exist_ok=True)
#         plot_file = f"{out_file_prefix}_boxplot.pdf"
#         plt.savefig(plot_file, bbox_inches='tight')
#         print(f'Saved file: {plot_file}')
#
#     plt.show()


def box_plot_subplots(data, x, y, ylabel, feature_filters, hue=None, y_min=None, y_max=None, rotate=0, palette=None,
                      color=None, hue_order=None, out_file_prefix=None, figsize=(6, 8), width=0.5, title='',
                      dodge=True, zero_line=False, legend='auto', edgecolor='black',
                      ft_filt_wise_1hot=None, bg_colors=None, orientation='vertical'):
    """
    Creates subplots for each `feature_filter`, with each subplot containing box plots.

    Parameters:
      - orientation: 'vertical' (default) or 'horizontal'. When 'horizontal', the boxplots are
                     drawn with numeric data on the x-axis and the subplots are arranged vertically,
                     sharing a common x-axis.
    """

    # Calculate counts for each feature_filter
    row_counts = [len(data[data['feature_filter'] == feature_filter]) for feature_filter in feature_filters]
    total_rows = sum(row_counts)

    if orientation == 'vertical':
        # For vertical boxplots: one row, multiple columns sharing the y-axis.
        ratios = [row_count / total_rows for row_count in row_counts]
        fig, axes = plt.subplots(1, len(feature_filters),
                                 figsize=figsize,
                                 gridspec_kw={'width_ratios': ratios}, sharey=True)
    else:
        # For horizontal boxplots: one column, multiple rows sharing the x-axis.
        ratios = [row_count / total_rows for row_count in row_counts]
        fig, axes = plt.subplots(len(feature_filters), 1,
                                 figsize=figsize,
                                 gridspec_kw={'height_ratios': ratios}, sharex=True)
        # Ensure axes is always iterable.
        if len(feature_filters) == 1:
            axes = [axes]

    for ax, feature_filter in zip(axes, feature_filters):
        subset = data[data['feature_filter'] == feature_filter].reset_index(drop=True)
        if subset.empty:
            continue

        # Determine unique models for the categorical axis (assumed to be in column 'Model')
        models = list(subset['Model'].unique())
        ax.set_axisbelow(True)  # Put background elements behind plot elements.
        if bg_colors:
            for i, model in enumerate(models):
                bg_color = bg_colors[0] if i % 2 == 0 else bg_colors[1]
                if orientation == 'vertical':
                    ax.axvspan(i - 0.5, i + 0.5, facecolor=bg_color, alpha=0.3, zorder=0)
                else:
                    ax.axhspan(i - 0.5, i + 0.5, facecolor=bg_color, alpha=0.3, zorder=0)

        # Plot boxplot. Swap x and y for horizontal orientation.
        if orientation == 'vertical':
            if color:
                box = sns.boxplot(
                    data=subset, x=x, y=y, dodge=dodge, width=width, ax=ax, linewidth=0.8,
                    color=color, boxprops={'edgecolor': edgecolor}
                )
            else:
                box = sns.boxplot(
                    data=subset, x=x, y=y, hue=hue, hue_order=hue_order, dodge=dodge,
                    width=width, palette=palette, ax=ax, linewidth=0.8, legend=legend,
                    boxprops={'edgecolor': edgecolor}
                )
        else:
            if color:
                box = sns.boxplot(
                    data=subset, x=y, y=x, dodge=dodge, width=width, ax=ax, linewidth=0.8,
                    color=color, boxprops={'edgecolor': edgecolor}
                )
            else:
                box = sns.boxplot(
                    data=subset, x=y, y=x, hue=hue, hue_order=hue_order, dodge=dodge,
                    width=width, palette=palette, ax=ax, linewidth=0.8, legend=legend,
                    boxprops={'edgecolor': edgecolor}
                )

        # Formatting adjustments based on orientation.
        if orientation == 'vertical':
            ax.set_xlabel('')
            ax.set_ylabel(ylabel, fontsize=14)
            ax.set_xticks(range(len(models)))
            ax.set_xticklabels(models, rotation=rotate, fontsize=12)
            ax.yaxis.grid(True, linestyle='--', color='grey', alpha=0.6, linewidth=0.6)
            if (y_min is not None) and (y_max is not None):
                ax.set_ylim(y_min, y_max)
            if ft_filt_wise_1hot:
                ax.axhline(y=ft_filt_wise_1hot[feature_filter], color='red', linestyle='--', linewidth=0.8, zorder=10)
            if zero_line:
                ax.axhline(y=0, color='red', linestyle='--', linewidth=0.8, zorder=10)
        else:
            ax.set_xlabel(ylabel, fontsize=14)  # Shared common x-axis label.
            ax.set_ylabel('')
            ax.set_yticks(range(len(models)))
            ax.set_yticklabels(models, rotation=rotate, fontsize=12)
            ax.xaxis.grid(True, linestyle='--', color='grey', alpha=0.6, linewidth=0.6)
            if (y_min is not None) and (y_max is not None):
                ax.set_xlim(y_min, y_max)
            if ft_filt_wise_1hot:
                ax.axvline(x=ft_filt_wise_1hot[feature_filter], color='red', linestyle='--', linewidth=0.8, zorder=10)
            if zero_line:
                ax.axvline(x=0, color='red', linestyle='--', linewidth=0.8, zorder=10)

        # Remove subplot legends and store handles if needed.
        if legend and box.get_legend() is not None:
            handles, labels = box.get_legend_handles_labels()
            ax.get_legend().remove()

    # Add a common legend if available.
    if legend and handles:
        fig.legend(handles, labels, loc='upper center', ncol=len(data[hue].unique()),
                   frameon=False, title=None, fontsize=12, bbox_to_anchor=(0.5, 1.02))

    plt.tight_layout()
    if out_file_prefix is not None:
        os.makedirs(os.path.dirname(out_file_prefix), exist_ok=True)
        plot_file = f"{out_file_prefix}_boxplot.pdf"
        plt.savefig(plot_file, bbox_inches='tight')
        print(f'Saved file: {plot_file}')

    plt.show()


def box_plot(data, x, y, ylabel, hue=None, y_min=None, y_max=None, rotate=0, palette=None, color=None,  hue_order=None,
             out_file_prefix=None,
             figsize=(6, 8), width=0.5, title='', n_cols=1, dodge=True, zero_line=False, legend='auto', edgecolor='black'):

    plt.figure(figsize=figsize)

    if color:
        sns.boxplot(data=data, x=x, y=y, dodge=dodge, width=width, boxprops={'facecolor': color, 'edgecolor':edgecolor}, linewidth=0.4, legend=legend)
    else:
        sns.boxplot(data=data, x=x, y=y, hue=hue, hue_order=hue_order, dodge=dodge, width=width, palette=palette,
                    linewidth=0.4, legend=legend)
    # Add labels and title
    # plt.xlabel('Models', fontsize=14)
    plt.ylabel(ylabel, fontsize=16)
    plt.xlabel('')
    # plt.title("Test Loss Distribution by Model and Rewired Status", fontsize=14)
    if (y_min is not None) and (y_max is not None):
        plt.ylim(y_min, y_max)
    # Add grid lines along the y-axis
    if zero_line:
        plt.axhline(y=0, color='red', linestyle='--', linewidth=0.8)
    else:
        plt.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.xticks(fontsize=14, rotation=rotate)
    plt.yticks(fontsize=14)

    if legend:
        if ylabel == 'Pearsons':
            plt.legend(loc="lower left", ncol=n_cols)
        else:
            plt.legend(loc="upper left", ncol=n_cols)

    plt.title(title)
    plt.tight_layout()
    if out_file_prefix is not None:
        plt.savefig(f'{out_file_prefix}_boxplot.pdf', bbox_inches='tight')
    # Show the plot

    plt.show()


def confidence_interval(std_dev, n, confidence_level=0.95):
    import scipy.stats as stats

    z_value = stats.norm.ppf(1 - (1 - confidence_level) / 2)
    se = std_dev / np.sqrt(n)

    return z_value * se


def compute_node_signed_strength(df, score_name):
    nodes = list(set(df['source'].unique()).union(set(df['target'].unique())))
    nodes.sort()
    positive_degree = {}
    negative_degree = {}

    for node in nodes:
        df_node = df[(df['source'] == node) | (df['target'] == node)]
        positive_degree[node] = df_node[df_node[score_name] > 0][score_name].sum()
        negative_degree[node] = df_node[df_node[score_name] < 0][score_name].sum()
    return positive_degree, negative_degree


def compute_node_strength(df, score_name):
    nodes = list(set(df['source'].unique()).union(set(df['target'].unique())))
    nodes.sort()
    degree = {}

    for node in nodes:
        df_node = df[(df['source'] == node) | (df['target'] == node)]
        degree[node] = df_node[score_name].sum()
    return degree, nodes


def wrapper_violin_plot_difference_in_degree_distribution(rewired_all_train_df, all_train_df, score_name, cell_line_2_idx, plot_file_prefix=None):
    # for each node compute its positive and negative weighted degree separately i.e., postive strength and negative strength resepectively.
    # Now for each node compute the difference between its positive strength in original vs randmoized network.

    edge_types = set(rewired_all_train_df['edge_type'].unique())
    plot_data = []
    for edge_type in edge_types:
        df1 = rewired_all_train_df[rewired_all_train_df['edge_type'] == edge_type]
        df2 = all_train_df[all_train_df['edge_type'] == edge_type]
        positive_degree_1, negative_degree_1 = compute_node_signed_strength(df1, score_name)
        positive_degree_2, negative_degree_2 = compute_node_signed_strength(df2, score_name)

        pos_diff = list({key: positive_degree_1[key] - positive_degree_2[key] for key in positive_degree_1.keys()}.values())
        neg_diff = list({key: negative_degree_1[key] - negative_degree_2[key] for key in negative_degree_1.keys()}.values())

        # Add data to the plot_data list
        plot_data.extend([{'edge_type': edge_type, 'degree_type': 'Positive Degree', 'diff': diff} for diff in pos_diff])
        plot_data.extend([{'edge_type': edge_type, 'degree_type': 'Negative Degree', 'diff': diff} for diff in neg_diff])

    # Convert the plot data to a DataFrame
    plot_df = pd.DataFrame(plot_data)

    #map cell line idx to name
    idx_2_cell_line = {idx:cell_line for (cell_line, idx) in cell_line_2_idx.items()}
    plot_df['cell_line_name'] =  plot_df['edge_type'].map(idx_2_cell_line)

    # Create the violin plot
    plt.figure(figsize=(6, 4))
    sns.violinplot(x='cell_line_name', y='diff', hue='degree_type', data=plot_df, split=True, inner='quart', density_norm='count', linewidth=0.4)

    plt.ylim(-10, 10)
    # Customize the plot
    plt.xlabel('Cell lines', fontsize=12)
    plt.ylabel('Difference in Weighted Degree', fontsize=12)
    # plt.title('Distribution of Differences in Degree by Cell line')
    plt.legend(loc='upper right')

    # Show and save the plot
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', linewidth=0.4, alpha=0.7)
    os.makedirs(os.path.dirname(plot_file_prefix), exist_ok=True)
    plt.savefig(f'{plot_file_prefix}_difference_in_degree_dist_with_rewired_plot.pdf', bbox_inches='tight')

    plt.show()
    plt.close()


def joint_plot(rewired, orig, score_name, idx_2_cell_line, min=None, max=None, plot_file_prefix=None):
    edge_types = set(rewired['edge_type'].unique())

    node_strength_dict={'node':[],'rewired_strength': [], 'orig_strength':[],'edge_type':[]}
    for i, edge_type in enumerate(edge_types):
        rewired_df = rewired[rewired['edge_type'] == edge_type]
        orig_df = orig[orig['edge_type'] == edge_type]
        rewired_deg, rewired_nodes = compute_node_strength(rewired_df, score_name)
        orig_deg, orig_nodes = compute_node_strength(orig_df, score_name)

        #keep the common nodes
        uncommon_nodes = set(orig_nodes).difference(set(rewired_nodes))
        assert len(uncommon_nodes) <5, print('too many nodes left out while randomizing')
        common_nodes = list(set(rewired_nodes).intersection(set(orig_nodes)))
        rewired_deg = {x:rewired_deg[x] for x in common_nodes}
        orig_deg = {x:orig_deg[x] for x in common_nodes}
        assert rewired_deg.keys() == orig_deg.keys()

        node_strength_dict['node'].extend(list(rewired_deg.keys()))
        node_strength_dict['rewired_strength'].extend(list(rewired_deg.values()))
        node_strength_dict['orig_strength'].extend(list(orig_deg.values()))
        node_strength_dict['edge_type'].extend([idx_2_cell_line[edge_type]]*len(orig_deg.keys()))
    #remove dipg_25 pick of hist is too high for this, which makes other edge type invisible
    node_strength_df = pd.DataFrame(node_strength_dict)

    # node_strength_df=node_strength_df[node_strength_df['edge_type']!='dipg25']
    # sns.jointplot(data=node_strength_df, x='orig_strength', y='rewired_strength', hue='edge_type', kind='scatter',
    #               ratio=10, height=4, marginal_kws={'bw_adjust': 0.5})


    def symmetric_log_transform(x):
        epsilon = 1e-6  # Small constant
        return np.sign(x) * np.log10(abs(x)+epsilon)

    node_strength_df['scaled_orig_strength'] = node_strength_df['orig_strength'].apply(symmetric_log_transform)
    node_strength_df['scaled_rewired_strength'] = node_strength_df['rewired_strength'].apply(symmetric_log_transform)

    sns.jointplot(
        data=node_strength_df,
        x='scaled_orig_strength',
        y='scaled_rewired_strength',
        hue='edge_type',
        kind='scatter',
        marginal_kws={'bw_adjust': 0.5},
        height=4,
        ratio=4
    )

    plt.xlabel('Node Strength in Original', fontsize=12)
    plt.ylabel('Node Strength in Randomized', fontsize=12)
    plt.legend(loc='upper left', fontsize="small" )
    plt.tight_layout()

    #
    # # Save the final figure
    os.makedirs(os.path.dirname(plot_file_prefix), exist_ok=True)
    plt.savefig(f'{plot_file_prefix}_difference_in_dist_with_rewired_plot.pdf', bbox_inches='tight')
    plt.show()
    plt.close()


def wrapper_plot_difference_in_degree_distribution(rewired_all_train_df, all_train_df, score_name, cell_line_2_idx, plot_file_prefix=None):
    # for each node compute its positive and negative weighted degree separately i.e., postive strength and negative strength resepectively.
    # Now for each node compute the difference between its positive strength in original vs randmoized network.
    # map cell line idx to name
    idx_2_cell_line = {idx: cell_line for (cell_line, idx) in cell_line_2_idx.items()}


    rewired_pos = rewired_all_train_df[rewired_all_train_df[score_name]>=0]
    rewired_neg = rewired_all_train_df[rewired_all_train_df[score_name]<0]

    orig_pos = all_train_df[all_train_df[score_name] >= 0]
    orig_neg = all_train_df[all_train_df[score_name] < 0]

    joint_plot(rewired_pos, orig_pos, score_name, idx_2_cell_line, min=0, plot_file_prefix = plot_file_prefix+'_pos_')
    joint_plot(rewired_neg, orig_neg, score_name, idx_2_cell_line, max=0, plot_file_prefix = plot_file_prefix+'_neg_')





def plot_dist(values, prefix='', out_dir=None):
    plt.clf()
    max = int(np.max(values))
    min = int(np.min(values))

    mean = round(np.mean(values), 2)
    std = round (np.std(values), 2)

    if min<0:
        bin_edges = list(np.arange(0, min, -15))
        bin_edges = list(set(bin_edges +  list(np.arange(0, max, 15)))) #remove duplicate 0s
        bin_edges.sort()
    else:
        bin_edges = list(np.arange(min, max, 15))
    # Create histogram with the specified bins
    counts, bins = np.histogram(values, bins=bin_edges)

    # Normalize counts to fractions
    fractions = counts / counts.sum()

    # Plot the binned bar plot with fractions
    plt.bar(bins[:-1], fractions, width=np.diff(bins), edgecolor="black", align="edge")

    # Set custom tick labels for bins
    # plt.xticks(bins[:-1] + np.diff(bins) / 2,  # Place ticks at bin centers
    #            [f'{int(bin_edges[i])}' for i in range(len(bin_edges) - 1)],
    #            rotation=45)
    plt.xticks(bin_edges[:-1], [f'{int(bins[i])}' for i in range(len(bins) - 1)],
               rotation=90, fontsize='small')

    # Add labels and title
    plt.xlabel('Score')
    plt.ylabel('Fraction of triplets')
    plt.title(prefix + f' Mean = {mean} and Standard Deviation = {std}')

    if out_dir is not None:
        filename = out_dir + f'{prefix}_score_distribution.pdf'
        plt.tight_layout()
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename)
    # Show plot
    plt.show()


def plot_double_dist(values_1, values_2, labels, prefix='', out_dir=None):

    '''
    Plot distribution of scores in values_1 and values_2 as violin plots
    :param values_1:
    :param values_2:
    :param labels:
    :param prefix:
    :param out_dir:
    :return:
    '''
    # max = int(np.max(np.concatenate((values_1, values_2))))
    # min = int(np.min(np.concatenate((values_1, values_2))))

    max=400
    min=-100

    mean_1 = round(np.mean(values_1), 2)
    std_1 = round(np.std(values_1), 2)
    mean_2 = round(np.mean(values_2), 2)
    std_2 = round(np.std(values_2), 2)
    skew_1 = round(skew(values_1), 2)
    skew_2 = round(skew(values_2), 2)

    bin_edges = list(np.arange(min, max, 15))
    plt.hist(values_1, bins=bin_edges, alpha=0.5, label=f'{labels[0]} (mean = {mean_1}, std = {std_1}, skew = {skew_1})', color='orange')
    plt.hist(values_2, bins=bin_edges, alpha=0.5, label=f'{labels[1]} (mean = {mean_2}, std = {std_2}, skew = {skew_2})', color='blue')
    # Add labels and title
    plt.xlabel('Score')
    plt.ylabel('number of triplets')
    plt.legend()
    plt.title(prefix)
    if out_dir is not None:
        filename = out_dir + f'{prefix}_dual_score_distribution.pdf'
        plt.tight_layout()
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename)
    # Show plot
    plt.show()


    #box_plot
    plt.figure(figsize=(4,6))
    sns.violinplot(data=[values_1, values_2])
    plt.xticks([0, 1], labels)
    plt.ylabel('Scores')
    plt.title(prefix)
    if out_dir is not None:
        filename = out_dir + f'{prefix}_dual_score_distribution_boxplot.pdf'
        plt.tight_layout()
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename)
    # plt.title('Comparison of Skewness')
    plt.show()



def plot_nodewise_train_test_score_dist(train_df, test_df, score_name, out_dir=None):

    train_edges = list(zip(train_df['source'],train_df['target'], train_df['edge_type'], train_df[score_name]))
    test_edges = list(zip(test_df['source'],test_df['target'],test_df['edge_type'], test_df[score_name]))

    # Function to collect scores for each (node, edge_type) pair
    def get_node_edge_type_scores(edges):
        node_edge_type_scores = {}
        for node1, node2,edge_type, score  in edges:
            if (node1, edge_type) not in node_edge_type_scores:
                node_edge_type_scores[(node1, edge_type)] = []
            if (node2, edge_type) not in node_edge_type_scores:
                node_edge_type_scores[(node2, edge_type)] = []
            node_edge_type_scores[(node1, edge_type)].append(score)
            node_edge_type_scores[(node2, edge_type)].append(score)
        return node_edge_type_scores

    # Get train and test scores for each (node, edge_type) pair
    train_node_edge_type_scores = get_node_edge_type_scores(train_edges)
    test_node_edge_type_scores = get_node_edge_type_scores(test_edges)

    # Prepare data for plotting
    data = []
    for (node, edge_type) in set(train_node_edge_type_scores.keys()).union(test_node_edge_type_scores.keys()):
        train_scores = train_node_edge_type_scores.get((node, edge_type), [])
        test_scores = test_node_edge_type_scores.get((node, edge_type), [])
        for score in train_scores:
            data.append((node, edge_type, 'Train', score))
        for score in test_scores:
            data.append((node, edge_type, 'Test', score))

    # Create a DataFrame for easy plotting with seaborn
    df = pd.DataFrame(data, columns=['Node', 'Edge Type', 'Set', 'Score'])

    # Compute median, mean, and standard deviation for each (node, edge_type, set) tuple
    stats_df = df.groupby(['Node', 'Edge Type', 'Set']).agg(
        median_score=('Score', 'median'),
        mean_score=('Score', 'mean'),
        std_score=('Score', 'std')
    ).reset_index()

    print("Statistics for each (node, edge_type, set) tuple:\n", stats_df)

    # Pivot the data to calculate differences between train and test for each (node, edge_type) pair
    pivoted_stats = stats_df.pivot(index=['Node', 'Edge Type'], columns='Set', values=['median_score', 'mean_score'])
    pivoted_stats.columns = ['_'.join(col).strip() for col in pivoted_stats.columns]
    pivoted_stats = pivoted_stats.reset_index()

    # Calculate the differences between train and test statistics for each (node, edge_type) pair
    pivoted_stats['median_diff'] = pivoted_stats['median_score_Train'] - pivoted_stats['median_score_Test']
    pivoted_stats['mean_diff'] = pivoted_stats['mean_score_Train'] - pivoted_stats['mean_score_Test']

    print("\nDifferences in median and mean between Train and Test for each (node, edge_type) pair:\n",
          pivoted_stats[['Node', 'Edge Type', 'median_diff', 'mean_diff']])



    return stats_df

