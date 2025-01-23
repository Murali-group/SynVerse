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






def compute_average_and_significance(df, n_runs=5):
    # Define a function to compute the Mann-Whitney U test for a group
    def compute_mannwhitney(group):
        stat, p_value = mannwhitneyu(group['test_loss'], group['test_loss_baseline'], alternative='less')
        return pd.Series({'stat': stat, 'p_value': p_value})

    # Group by and compute aggregation metrics
    aggregated_results = df.groupby(['drug_features', 'cell_features', 'feature_filter', 'Model']).agg(
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
        df.groupby(['drug_features', 'cell_features', 'feature_filter', 'Model'])
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

    # # Calculate global y-axis limits. Make ylims divisible by 5
    # y_min = min(df_1hot_diff_avg[mean_col] - df_1hot_diff_avg[err_col])
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



def set_model_names(df):
    # Create x-tick labels combining 'drug_features' and 'cell_features'
    df['Model'] = df['drug_features'] + " + " + df['cell_features']
    df['Model'] = df['Model'].astype(str).apply(lambda x: model_name_mapping.get(x, x))

    # remove model 'One hot (AE)'
    df = df[df['Model'] != 'One hot (AE)']

    # remove a few feature combo if present. Following remove model where auto-encoder used on one-hot without standardization.
    df = df[~((df['drug_features'] == 'd1hot_comp_True') | (
            df['cell_features'] == 'c1hot_comp_True'))]

    return df

def pair_plot(df_all, metric, out_file_prefix):

    models = df_all['Model'].unique()


    # Create a grid of 4x4 subplots
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))  # Adjust figure size as needed
    axes = axes.flatten()  # Flatten to easily iterate through axes

    max_loss = max(df_all['test_loss'].max(), df_all['test_loss_baseline'].max())
    min_val = 5

    for (i,model) in enumerate(models):
        df = df_all[df_all['Model'] == model]
        ax = axes[i]
        # Create the pair plot
        # colors = ['red' if x > y else 'blue' for x, y in zip( df['test_loss'], df['test_loss_baseline'])]

        sns.scatterplot(
            x='test_loss',
            y='test_loss_baseline',
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
    out_file = f'{out_file_prefix}_pairplot.pdf'
    plt.savefig(out_file, bbox_inches='tight')
    print(f'saving file at {out_file}')
    plt.show()



def wrapper_plot_compare_with_1hot(df_MSE, title, out_file_prefix):

    #compute RMSE from MSE:
    df_RMSE= copy.deepcopy(df_MSE)
    for  metric in ['test_loss', 'train_loss','val_loss']:
        df_RMSE[metric] = np.sqrt(df_MSE[metric])

    data_dict = {'Root Mean Squared Error (RMSE)': df_RMSE, 'Mean Squared Error (MSE)': df_MSE}

    #compute the difference between the model's result and corresponding 1 hot encoding
    #plot MSE
    for measure in data_dict:
        file_name_suitable_measure = measure.split(' ')[-1].replace('(','').replace(')','')#metrci name, MSE or RMSE

        df_1hot_diff = compute_difference_with_1hot(data_dict[measure])
        df_1hot_diff = set_model_names(df_1hot_diff) #give model name from features.

        #draw a pairplot to show difference between
        df_1hot_diff_avg = compute_average_and_significance(df_1hot_diff)

        # Sort and save DataFrame according to the order of models in model_name_mapping
        df_1hot_diff['Model'] = pd.Categorical(df_1hot_diff['Model'], categories=model_name_mapping.values(),ordered=True)
        df_1hot_diff = df_1hot_diff.sort_values('Model')
        df_1hot_diff.to_csv(f'{out_file_prefix}_{file_name_suitable_measure}_performance_compared_to_baseline.tsv', sep='\t')


        #sort model names
        df_1hot_diff_avg['Model'] = pd.Categorical(df_1hot_diff_avg['Model'], categories=model_name_mapping.values(), ordered=True)
        df_1hot_diff_avg = df_1hot_diff_avg.sort_values('Model')
        df_1hot_diff_avg.to_csv(f'{out_file_prefix}_{file_name_suitable_measure}_aggreagred_performance.tsv', sep='\t')


        #pair plot for comparing each modelw ith baseline across each individual run
        pair_plot(df_1hot_diff, file_name_suitable_measure, out_file_prefix=f'{out_file_prefix}_{file_name_suitable_measure}')

        #bar plot for showing average MSE/RMSE of each model, showing performance improvement over baseline with color.
        plot_diff(df_1hot_diff_avg, measure, title, metric ='test_loss', yerr='std', color_on = 'diff_mean', out_file_prefix=f'{out_file_prefix}_{file_name_suitable_measure}')
        plot_diff(df_1hot_diff_avg, measure, title, metric ='test_loss', yerr='std', color_on = 'is_significant', out_file_prefix=f'{out_file_prefix}_{file_name_suitable_measure}')


    print(title)


def wrapper_plot_compare_rewired(result_df, rewired_result_df, out_file_prefix):

        df = pd.concat([result_df, rewired_result_df], axis=0)
        df = set_model_names(df)
        df['test_loss_RMSE'] = np.sqrt(df['test_loss'])


        #keeps models  which I ran on rewired network
        rewired_model_names = df[df['rewired']==True]['Model'].unique()
        df = df[df['Model'].isin(rewired_model_names)]

        #sort model names
        df['Model'] = pd.Categorical(df['Model'], categories=model_name_mapping.values(),ordered=True)
        df = df.sort_values('Model')

        #modify model name to look good on plot
        df['Model'] = df['Model'].str.replace(r'\(', r'\n(', regex=True)

        def box_lot(y, metric):
            #plot test MSE loss
            plt.figure(figsize=(6, 4))
            sns.boxplot(data=df, x='Model', y=y, hue='rewired', dodge=True,  width=0.5,   palette="Set2", linewidth=0.4)
            # Add labels and title
            plt.xlabel("Models", fontsize=12)
            if metric == 'MSE':
                plt.ylabel("Mean Squared Error (MSE)", fontsize=12)
            if metric == 'RMSE':
                plt.ylabel("Root Mean Squared Error (RMSE)", fontsize=12)
            # plt.title("Test Loss Distribution by Model and Rewired Status", fontsize=14)
            # Add legend
            plt.ylim(0, 20)
            # Add grid lines along the y-axis
            plt.grid(axis='y', linestyle='--', linewidth=0.4, alpha=0.7)
            plt.xticks(fontsize=10)


            # Add vertical lines between models
            # unique_models = df['Model'].unique()
            # for i in range(1, len(unique_models)):
            #     plt.axvline(i - 0.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)


            plt.legend(title="Rewired", loc="lower right")
            plt.savefig(f'{out_file_prefix}_rewired_{metric}.pdf', bbox_inches='tight')
            # Show the plot
            plt.tight_layout()
            plt.show()
        # box_lot('test_loss', 'MSE')
        df_1 = df.groupby(['Model','rewired']).agg({'test_loss_RMSE': 'mean'})
        print(df_1)
        box_lot('test_loss_RMSE', 'RMSE')




def main():
    score_names = ['S_mean_mean', 'synergy_loewe_mean']
    split_types = ['leave_comb', 'leave_drug', 'leave_cell_line', 'random']

    for score_name in score_names:
        result_dir = f'/home/grads/tasnina/Projects/SynVerse/outputs/k_0.05_{score_name}'
        for split_type in split_types:
            # plot for comparing models with each other. Also compare with one hot based model i.e., basleine
            result_file = f'output_{split_type}.tsv'
            result_file_path = os.path.join(result_dir, result_file)
            if not os.path.exists(result_file_path):
                print(f'file {result_file} does not exist. Continuing to next file.')
                continue
            result_df = pd.read_csv(result_file_path, sep='\t', index_col=None)
            wrapper_plot_compare_with_1hot(result_df, title=split_type, out_file_prefix = f'{result_dir}/{score_name}_{split_type}')

            # plot for comparing models trained on original vs. rewired networks
            rewired_net_result_file = f'output_{split_type}_rewired.tsv'
            rewired_result_file_path = os.path.join(result_dir, rewired_net_result_file)
            if not os.path.exists(rewired_result_file_path):
                print(f'file {rewired_result_file_path} does not exist. Continuing to next file.')
                continue
            rewired_result_df = pd.read_csv(rewired_result_file_path, sep='\t', index_col=None)
            wrapper_plot_compare_rewired(result_df, rewired_result_df, out_file_prefix= f'{result_dir}/{score_name}_{split_type}')



    # barplot_model_comparison_with_deepsynergy("/home/grads/tasnina/Projects/SynVerse/inputs/existing_model_performance.tsv")
    # scatter_plot_model_comparison_with_deepsynergy("/home/grads/tasnina/Projects/SynVerse/inputs/existing_model_performance.tsv")


if __name__ == '__main__':
    main()