import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def compute_node_strength(df, score_name):
    nodes = set(df['source'].unique()).union(set(df['target'].unique()))
    positive_degree = {}
    negative_degree = {}

    for node in nodes:
        df_node = df[(df['source'] == node) | (df['target'] == node)]
        positive_degree[node] = df_node[df_node[score_name] > 0][score_name].sum()
        negative_degree[node] = df_node[df_node[score_name] < 0][score_name].sum()
    return positive_degree, negative_degree


def plot_difference_in_degree_distribution(dict1, dict2):
    if dict1.keys() != dict2.keys():
        raise ValueError("The dictionaries do not have the same keys!")

        # Compute the differences
    differences = {key: dict1[key] - dict2[key] for key in dict1.keys()}

    # Extract the differences as a list
    diff_values = list(differences.values())

    # Plot the distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(diff_values,  bins=50, color='blue')
    plt.title('Distribution of differences between node degrees')
    plt.xlabel('Difference')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

def wrapper_plot_difference_in_degree_distribution(rewired_all_train_df, all_train_df, score_name, plot_file_prefix=None):
    # for each node compute its positive and negative weighted degree separately i.e., postive strength and negative strength resepectively.
    # Now for each node compute the difference between its positive strength in original vs randmoized network.

    edge_types = set(rewired_all_train_df['edge_type'].unique())

    for edge_type in edge_types:
        df1 = rewired_all_train_df[rewired_all_train_df['edge_type'] == edge_type]
        df2 = all_train_df[all_train_df['edge_type'] == edge_type]
        positive_degree_1, negative_degree_1 = compute_node_strength(df1, score_name)
        positive_degree_2, negative_degree_2 = compute_node_strength(df2, score_name)

        pos_diff = list({key: positive_degree_1[key] - positive_degree_2[key] for key in positive_degree_1.keys()}.values())
        neg_diff = list({key: negative_degree_1[key] - negative_degree_2[key] for key in negative_degree_1.keys()}.values())
        plt.figure(figsize=(4, 6))

        sns.histplot(pos_diff, bins=50, color='blue', label='Positive Degree')
        sns.histplot(neg_diff, bins=50, color='yellow', label='Negative Degree')

        plt.xlabel('Difference in Weighted Degree')
        plt.ylabel('Number of Nodes')
        plt.title(f'{edge_type}')
        plt.legend(loc='upper right')
        plt.savefig(f'{plot_file_prefix}_{edge_type}', bbox_inches='tight')
        plt.show()
        plt.close()

        #
        # plot_difference_in_degree_distribution(positive_degree_1, positive_degree_2)
        # plot_difference_in_degree_distribution(negative_degree_1, negative_degree_2)





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