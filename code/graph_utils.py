import os.path

import pandas as pd

from evaluation.split import *
import bct
import numpy as np
from tqdm import tqdm
from sklearn.utils import check_random_state

'''
Reference URL: https://github.com/fmilisav/milisav_strength_nulls
'''
import bct
import math
import numpy as np
from sklearn.utils import check_random_state


def strength_preserving_rand_rs(A,
                                rewiring_iter=10, sort_freq=1,
                                R=None, connected=None,
                                seed=None):
    """
    Degree- and strength-preserving randomization of
    undirected, weighted adjacency matrix A

    Parameters
    ----------
    A : (N, N) array-like
        Undirected weighted adjacency matrix
    rewiring_iter : int, optional
        Rewiring parameter. Default = 10.
	    Each edge is rewired approximately rewiring_iter times.
    sort_freq : float, optional
        Frequency of weight sorting. Must be between 0 and 1.
        If 1, weights are sorted at every iteration.
        If 0.1, weights are sorted at every 10th iteration.
        A higher value results in a more accurate strength sequence match.
        Default = 1.
    R : (N, N) array-like, optional
        Pre-randomized adjacency matrix.
        If None, a rewired adjacency matrix is generated using the
        Maslov & Sneppen algorithm.
        Default = None.
    connected: bool, optional
        Whether to ensure connectedness of the randomized network.
        By default, this is inferred from data.
    seed: float, optional
        Random seed. Default = None.

    Returns
    -------
    B : (N, N) array-like
        Randomized adjacency matrix

    Notes
    -------
    Uses Maslov & Sneppen rewiring to produce a
    surrogate adjacency matrix, B, with the same
    size, density, and degree sequence as A.
    The weights are then permuted to optimize the
    match between the strength sequences of A and B.

    This function is adapted from a function written in MATLAB
    by Mika Rubinov (https://sites.google.com/site/bctnet/home).
    It was adapted to positive structural connectivity networks
    from an algorithm originally developed for
    signed functional connectivity networks.

    References
    -------
    Maslov & Sneppen (2002) Specificity and stability in
    topology of protein networks. Science.
    Rubinov & Sporns (2011) Weight-conserving characterization of
    complex functional brain networks. Neuroimage.
    """

    A = A.copy()
    try:
        A = np.asarray(A)
    except TypeError as err:
        msg = ('A must be array_like. Received: {}.'.format(type(A)))
        raise TypeError(msg) from err

    if sort_freq > 1 or sort_freq <= 0:
        msg = ('sort_freq must be between 0 and 1. '
               'Received: {}.'.format(sort_freq))
        raise ValueError(msg)

    rs = check_random_state(seed)

    n = A.shape[0]

    # clearing diagonal
    np.fill_diagonal(A, 0)

    if R is None:
        # ensuring connectedness if the original network is connected
        if connected is None:
            connected = False if bct.number_of_components(A) > 1 else True

        # Maslov & Sneppen rewiring
        if connected:
            R = bct.randmio_und_connected(A, rewiring_iter, seed=seed)[0]
        else:
            R = bct.randmio_und(A, rewiring_iter, seed=seed)[0]

    B = np.zeros((n, n))
    s = np.sum(A, axis=1)  # strengths of A
    sortAvec = np.sort(A[np.triu(A, k=1) > 0])  # sorted weights vector
    x, y = np.nonzero(np.triu(R, k=1))  # weights indices

    E = np.outer(s, s)  # expected weights matrix

    if sort_freq == 1:
        for i in range(len(sortAvec) - 1, -1, -1):
            sort_idx = np.argsort(E[x, y])  # indices of x and y that sort E

            r = math.ceil(rs.rand() * i)
            r_idx = sort_idx[r]  # random index of sorted expected weight matrix

            # assigning corresponding sorted weight at this index
            B[x[r_idx], y[r_idx]] = sortAvec[r]

            # radjusting the expected weight probabilities of
            # the node indexed in x
            f = 1 - sortAvec[r] / s[x[r_idx]]
            E[x[r_idx], :] *= f
            E[:, x[r_idx]] *= f

            # radjusting the expected weight probabilities of
            # the node indexed in y
            f = 1 - sortAvec[r] / s[y[r_idx]]
            E[y[r_idx], :] *= f
            E[:, y[r_idx]] *= f

            # readjusting residual strengths of nodes indexed in x and y
            s[x[r_idx]] -= sortAvec[r]
            s[y[r_idx]] -= sortAvec[r]

            # removing current weight
            x = np.delete(x, r_idx)
            y = np.delete(y, r_idx)
            sortAvec = np.delete(sortAvec, r)
    else:
        sort_period = round(1 / sort_freq)  # sorting period
        for i in range(len(sortAvec) - 1, -1, -sort_period):
            sort_idx = np.argsort(E[x, y])  # indices of x and y that sort E

            r = rs.choice(i, min(i, sort_period), replace=False)
            r_idx = sort_idx[r]  # random indices of sorted expected weight matrix

            # assigning corresponding sorted weights at these indices
            B[x[r_idx], y[r_idx]] = sortAvec[r]

            xy_nodes = np.append(x[r_idx], y[r_idx])  # randomly indexed nodes
            xy_nodes_idx = np.unique(xy_nodes)  # randomly indexed nodes' indices

            # nodal cumulative weights
            accumWvec = np.bincount(xy_nodes,
                                    weights=sortAvec[np.append(r, r)],
                                    minlength=n)

            # readjusting expected weight probabilities
            F = 1 - accumWvec[xy_nodes_idx] / s[xy_nodes_idx]
            F = F[:, np.newaxis]

            E[xy_nodes_idx, :] *= F
            E[:, xy_nodes_idx] *= F.T

            # readjusting residual strengths of nodes indexed in x and y
            s[xy_nodes_idx] -= accumWvec[xy_nodes_idx]

            # removing current weight
            x = np.delete(x, r_idx)
            y = np.delete(y, r_idx)
            sortAvec = np.delete(sortAvec, r)

    B += B.T

    return B

def strength_preserving_rand_sa(A, rewiring_iter = 10,
                                nstage = 100, niter = 10000,
                                temp = 1000, frac = 0.5,
                                R = None, connected = None,
                                verbose = False, seed = None):
    """
    Degree- and strength-preserving randomization of
    undirected, weighted adjacency matrix A

    Parameters
    ----------
    A : (N, N) array-like
        Undirected weighted adjacency matrix
    rewiring_iter : int, optional
        Rewiring parameter. Default = 10.
        Each edge is rewired approximately rewiring_iter times.
    nstage : int, optional
        Number of annealing stages. Default = 100.
    niter : int, optional
        Number of iterations per stage. Default = 10000.
    temp : float, optional
        Initial temperature. Default = 1000.
    frac : float, optional
        Fractional decrease in temperature per stage. Default = 0.5.
    R : (N, N) array-like, optional
        Pre-randomized adjacency matrix.
        If None, a rewired adjacency matrix is generated using the
        Maslov & Sneppen algorithm.
        Default = None.
    connected: bool, optional
        Whether to ensure connectedness of the randomized network.
        By default, this is inferred from data.
    verbose: bool, optional
        Whether to print status to screen at the end of every stage.
        Default = False.
    seed: float, optional
        Random seed. Default = None.

    Returns
    -------
    B : (N, N) array-like
        Randomized adjacency matrix
    energymin : float
        Minimum energy obtained by annealing

    Notes
    -------
    Uses Maslov & Sneppen rewiring model to produce a
    surrogate adjacency matrix, B, with the same
    size, density, and degree sequence as A.
    The weights are then permuted to optimize the
    match between the strength sequences of A and B
    using simulated annealing.

    This function is adapted from a function written in MATLAB
    by Richard Betzel.

    References
    -------
    Misic, B. et al. (2015) Cooperative and Competitive Spreading Dynamics
    on the Human Connectome. Neuron.
    """

    try:
        A = np.asarray(A)
    except TypeError as err:
        msg = ('A must be array_like. Received: {}.'.format(type(A)))
        raise TypeError(msg) from err

    if frac > 1 or frac <= 0:
        msg = ('frac must be between 0 and 1. '
               'Received: {}.'.format(frac))
        raise ValueError(msg)

    rs = check_random_state(seed)

    n = A.shape[0]
    s = np.sum(A, axis = 1) #strengths of A

    #Maslov & Sneppen rewiring
    if R is None:
        #ensuring connectedness if the original network is connected
        if connected is None:
            connected = False if bct.number_of_components(A) > 1 else True
        if connected:
            B = bct.randmio_und_connected(A, rewiring_iter, seed=seed)[0]
        else:
            B = bct.randmio_und(A, rewiring_iter, seed=seed)[0]
    else:
        B = R.copy()

    u, v = np.triu(B, k = 1).nonzero() #upper triangle indices
    wts = np.triu(B, k = 1)[(u, v)] #upper triangle values
    m = len(wts)
    sb = np.sum(B, axis = 1) #strengths of B

    energy = np.mean((s - sb)**2)

    energymin = energy
    wtsmin = wts.copy()

    if verbose:
        print('\ninitial energy {:.5f}'.format(energy))

    for istage in tqdm(range(nstage), desc='annealing progress'):
        naccept = 0
        for (e1, e2), prob in zip(rs.randint(m, size=(niter, 2)),
                                  rs.rand(niter)
                                  ):

            #permutation
            a, b, c, d = u[e1], v[e1], u[e2], v[e2]
            wts_change = wts[e1] - wts[e2]
            delta_energy = (2 * wts_change *
                            (2 * wts_change +
                             (s[a] - sb[a]) +
                             (s[b] - sb[b]) -
                             (s[c] - sb[c]) -
                             (s[d] - sb[d])
                             )
                            )/n

            #permutation acceptance criterion
            if (delta_energy < 0 or prob < np.e**(-(delta_energy)/temp)):

                sb[[a, b]] -= wts_change
                sb[[c, d]] += wts_change
                wts[[e1, e2]] = wts[[e2, e1]]

                energy = np.mean((sb - s)**2)

                if energy < energymin:
                    energymin = energy
                    wtsmin = wts.copy()
                naccept = naccept + 1

        #temperature update
        temp = temp*frac
        if verbose:
            print('\nstage {:d}, temp {:.5f}, best energy {:.5f}, '
                  'frac of accepted moves {:.3f}'.format(istage, temp,
                                                         energymin,
                                                         naccept/niter))

    B = np.zeros((n, n))
    B[(u, v)] = wtsmin
    B = B + B.T

    return B, energymin


def dataframe_to_numpy(df, score_name):
    """
    Converts a pandas DataFrame to a numpy adjacency matrix.

    Args:
        df (pd.DataFrame): DataFrame with 'source', 'target', and 'score_name' columns.

    Returns:
        tuple: (adj_matrix, node_to_idx) where adj_matrix is a numpy array,
               and node_to_idx is a mapping of nodes to indices.
    """
    # Get unique nodes and map them to indices
    nodes = pd.unique(df[['source', 'target']].values.ravel())
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}

    # Initialize adjacency matrix
    N = len(nodes)
    adj_matrix = np.zeros((N, N))

    # Populate undirected adjacency matrix
    for _, row in df.iterrows():
        src_idx = node_to_idx[row['source']]
        tgt_idx = node_to_idx[row['target']]
        adj_matrix[src_idx, tgt_idx] = row[score_name]
        adj_matrix[tgt_idx, src_idx] = row[score_name]  # Ensure symmetry

    return adj_matrix, node_to_idx


def numpy_to_dataframe(adj_matrix, node_to_idx, score_name):
    """
    Converts a numpy adjacency matrix back to a pandas DataFrame.

    Args:
        adj_matrix (np.ndarray): Numpy adjacency matrix.
        node_to_idx (dict): Mapping of nodes to indices.

    Returns:
        pd.DataFrame: DataFrame with 'source', 'target', and 'score_name' columns.
    """
    # Reverse mapping from indices to nodes
    idx_to_node = {idx: node for node, idx in node_to_idx.items()}

    # Collect edges from adjacency matrix
    edges = []
    N = adj_matrix.shape[0]
    for i in range(N):
        for j in range(i + 1, N):  # Only upper triangle to avoid duplicates
            if adj_matrix[i, j] != 0:
                edges.append({
                    'source': idx_to_node[i],
                    'target': idx_to_node[j],
                    score_name: adj_matrix[i, j]
                })

    # Create DataFrame from edges
    return pd.DataFrame(edges)


def rewire(df, score_name, method='SA'):
    '''keeping the node degree intact, rewire the edges. We are shuffling pairs of edges. However, as we have to consider the score
    of the new edges, we  pair up two synergistic
    edges or two nonsynergistic edges when we shuffle. '''
    print(df.head(5))
    edge_types = set(df['edge_type'].unique())

    rewired_df = pd.DataFrame()

    for edge_type in edge_types:
        df_edge = df[df['edge_type'] == edge_type][['source', 'target', score_name]]
        A, node_2_idx = dataframe_to_numpy(df_edge, score_name)

        if method == 'SA': #simmulated annealing
            B, _ = strength_preserving_rand_sa (A)
        elif method =='RS': #rubinov and sporns
            B = strength_preserving_rand_rs(A)

        rewired_df_edge = numpy_to_dataframe(B, node_2_idx, score_name)
        rewired_df_edge['edge_type'] = edge_type
        rewired_df = pd.concat([rewired_df, rewired_df_edge], axis=0)
    return rewired_df



def get_rewired_train_val (all_train_df, score_name, method, split_type, val_frac, out_dir, force_run):
    rewired_train_file = out_dir + f'all_train_rewired_{method}.tsv'
    if (not os.path.exists(rewired_train_file)) | force_run:
        rewired_train_df = rewire(all_train_df, score_name, method=method)
        rewired_train_df.to_csv(rewired_train_file, sep='\t', index=False)
    else:
        rewired_train_df = pd.read_csv(rewired_train_file, sep='\t', index_col=None)

    split_type_map = {'random': 'random', 'leave_comb': 'edge', 'leave_drug': 'node', 'leave_cell_line': 'edge_type'}
    train_idx, val_idx = split_train_test(rewired_train_df, split_type_map[split_type], val_frac)

    return rewired_train_df, {0:train_idx}, {0:val_idx}


