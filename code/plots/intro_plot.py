import os.path
import matplotlib.cm as cm

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
import seaborn as sns
from matplotlib.patches import Patch  # Import at the top

def barplot_model_comparison_with_deepsynergy(filename, metric, out_file):
    df_all = pd.read_csv(filename, sep='\t')[['Model', f'DeepSynergy_{metric}', f'Own_{metric}', 'Dataset_Score']]
    df = df_all[df_all['Model']!='DeepSynergy (2017)']
    deepsynergy_performance = list(df_all[df_all['Model']=='DeepSynergy (2017)'][f'Own_{metric}'])[0]
    # Define color mapping
    # colors = {'Own': 'royalblue', 'DeepSynergy': 'tomato'}
    # viridis = cm.get_cmap('viridis')
    # own_color = viridis(0.4)  # A mid-range viridis color for 'Own' model
    # deepsynergy_color = viridis(0.9)  # A different viridis color for 'DeepSynergy'
    # colors = {'Own': own_color, 'DeepSynergy': deepsynergy_color}

    # palette = sns.cubehelix_palette(start=0.3, hue=1,
    #                                 gamma=0.4, dark=0.1, light=0.5,
    #                                 rot=-0.6, reverse=True, n_colors=2)
    # colors = {'Own': palette[0], 'DeepSynergy': palette[1]}

    colors = {'Own': '#452c63', 'DeepSynergy':'#90EE90'}

    # Define hatch patterns for Dataset_Score
    unique_datasets = df['Dataset_Score'].unique()
    # hatch_patterns = {dataset: pattern for dataset, pattern in
    #                   zip(unique_datasets, ['/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*'])}

    hatch_patterns = {dataset: pattern for dataset, pattern in
                      zip(unique_datasets, ['////', '----', 'xxxx', 'o', 'O', '.', ])}

    # Set up plot
    fig, ax = plt.subplots(figsize=(4.5, 5))

    # Bar width and positions
    x = np.arange(len(df))
    width = 0.4  # Width of each bar

    # Plot bars with hatches
    bars1 = ax.bar(x - width / 2, df[f'Own_{metric}'], width, label=f'Own', color=colors['Own'],
                   alpha=0.4,
                   hatch=[hatch_patterns[ds] for ds in df['Dataset_Score']],
                   )
    bars2 = ax.bar(x + width / 2, df[f'DeepSynergy_{metric}'], width, label='DeepSynergy', color=colors['DeepSynergy'],
                   alpha=0.5,
                   hatch=[hatch_patterns[ds] for ds in df['Dataset_Score']],
                   )

    # Plot a solid line for DeepSynergy's performance
    ax.axhline(y=deepsynergy_performance, color='orange', linestyle='--', linewidth=1)
    # X-axis labels and ticks
    ax.set_xticks(x)
    # ax.set_xticklabels(df['Model'].replace('(','\n('), rotation=45, ha='right')
    ax.set_xticklabels(df['Model'].str.replace(r'\(', r'\n(', regex=True), multialignment='center')

    # Labels and title
    ax.set_ylabel(metric)
    if metric=='Pearsons':
        ax.set_ylim(0, 1)

    # # ax.set_title(f'Model Comparison with DeepSynergy - {metric}')
    # custom_legend = [plt.Rectangle((0, 0), 1, 1, color=colors['Own'], alpha=0.7, label='Own'),
    #                  plt.Rectangle((0, 0), 1, 1, color=colors['DeepSynergy'], alpha=0.7,label='DeepSynergy'),
    #                  ]  # Line legend
    #
    # ax.legend(handles=custom_legend, loc='upper left')

    # ✅ Create a legend for colors (Own vs. DeepSynergy)
    color_legend = [plt.Rectangle((0, 0), 1, 0.8, color=colors['Own'],alpha=0.5, label='Own'),
                    plt.Rectangle((0, 0), 1, 0.8, color=colors['DeepSynergy'], alpha=0.5, label='DeepSynergy')]

    # legend1 = ax.legend(handles=color_legend, loc='upper right', ncol=2)

    # ✅ Create a legend for hatches (Dataset Scores)
    hatch_legend = [Patch(facecolor='none', edgecolor='gray', hatch=hatch_patterns[ds], label=f'{ds}')
                    for ds in unique_datasets]

    # legend2 = ax.legend(handles=hatch_legend, loc='upper left', frameon=True, ncol=3)
    combined_legend = color_legend + hatch_legend
    ax.legend(handles=combined_legend, loc='upper left', ncol=2, columnspacing=0.4, handletextpad=0.3)

    # ax.add_artist(legend1)  # Ensures the first legend stays
    # ax.add_artist(legend2)  # Adds the second legend

    plt.tight_layout()
    plt.savefig(out_file)
    plt.show()


file_name = "/home/grads/tasnina/Projects/SynVerse/inputs/existing_model_MSE_Pearsons.tsv"
metrics = ['Pearsons', 'RMSE']
for metric in metrics:
    out_file = f"/home/grads/tasnina/Projects/SynVerse/inputs/compare_with_deepsynergy_{metric}.pdf"
    barplot_model_comparison_with_deepsynergy(file_name, metric, out_file)
