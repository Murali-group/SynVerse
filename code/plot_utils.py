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
    plt.xticks(bin_edges[:-1], [f'{int(bins[i])}' for i in range(len(bins) - 1)], rotation=90)

    # Add labels and title
    plt.xlabel('Values')
    plt.ylabel('Fraction')
    plt.title(prefix + ' distribution')

    filename = out_dir + f'{prefix}_score_distribution.pdf'
    plt.tight_layout()
    plt.savefig(filename)
    # Show plot
    plt.show()
