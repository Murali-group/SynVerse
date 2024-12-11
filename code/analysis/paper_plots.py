import os.path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from adjustText import adjust_text

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

    # # Plotting the bars
    # bars1 = ax.bar([i - bar_width / 2 for i in x], df['Own'], bar_width, label='Own Performance', color='green')
    # bars2 = ax.bar([i + bar_width / 2 for i in x], df['DeepSynergy'], bar_width,
    #                label='Reported DeepSynergy Performance', color='blue')

    import matplotlib.cm as cm
    from matplotlib.colors import ListedColormap
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


barplot_model_comparison_with_deepsynergy("/home/grads/tasnina/Projects/SynVerse/inputs/existing_model_performance.tsv")

# scatter_plot_model_comparison_with_deepsynergy("/home/grads/tasnina/Projects/SynVerse/inputs/existing_model_performance.tsv")

