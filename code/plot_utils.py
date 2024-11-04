import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_dist(values, prefix='', out_dir=None):
    plt.clf()
    max = int(np.max(values))
    min = int(np.min(values))

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
               rotation=90)

    # Add labels and title
    plt.xlabel('Values')
    plt.ylabel('Fraction')
    plt.title(prefix + ' distribution')

    filename = out_dir + f'{prefix}_score_distribution.pdf'
    plt.tight_layout()
    plt.savefig(filename)
    # Show plot
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

    # Plotting boxplots for each (node, edge_type) pair, with separate colors for train and test
    # plt.figure(figsize=(16, 8))
    # sns.boxplot(x='Node', y='Score', hue='Set', data=df, col='Edge Type')
    # plt.title("Distribution of Scores per (Node, Edge Type) Pair (Train vs. Test)")
    # plt.xlabel("Node")
    # plt.ylabel("Score")
    #
    # plt.legend(title="split")
    # plt.tight_layout()
    # plt.savefig(out_dir + f'{score_name}_nodewise_test_train_scores.pdf')
    # plt.show()

    return stats_df