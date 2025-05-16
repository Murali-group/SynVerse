import os.path

import pandas as pd

from split import *
from tqdm import tqdm
from utils import *
import matplotlib.pyplot as plt

'''
Reference URL: https://github.com/fmilisav/milisav_strength_nulls
'''
import bct
import math
import numpy as np
from sklearn.utils import check_random_state




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

def degree_preserving_rand_sm(A, rewiring_iter = 10,
                                frac = 0.5,
                                R = None, connected = None,
                                seed = None):

    try:
        A = np.asarray(A)
    except TypeError as err:
        msg = ('A must be array_like. Received: {}.'.format(type(A)))
        raise TypeError(msg) from err

    if frac > 1 or frac <= 0:
        msg = ('frac must be between 0 and 1. '
               'Received: {}.'.format(frac))
        raise ValueError(msg)

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
    #check for symmetry
    # symmetric = is_symmetric(B)
    # print (f'SM returned symmetric array {symmetric}')
    return B


def strength_preserving_rand_sa_signed(A, rewiring_iter = 2,
                                       nstage = 100, niter = 10000,
                                       temp = 1000, frac = 0.5,
                                       energy_type = 'sse', energy_func = None,
                                       R = None, verbose = False,
                                       seed = None):
    """
    Degree- and strength-preserving randomization of
    undirected, weighted, signed adjacency matrix A

    Parameters
    ----------
    A : (N, N) array-like
        Undirected weighted signed adjacency matrix
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
    energy_type: str, optional
        Energy function to minimize. Can be either:
            'sse': Sum of squares between strength sequence vectors
                   of the original network and the randomized network
            'max': The single largest value
                   by which the strength sequences deviate
            'mae': Mean absolute error
            'mse': Mean squared error
            'rmse': Root mean squared error
        Default = 'sse'.
    energy_func: callable, optional
        Callable with two positional arguments corresponding to
        two strength sequence numpy arrays that returns an energy value.
        Overwrites “energy_type”.
        See “energy_type” for specifying a predefined energy type instead.
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
    energymin : dictionary
        Minimum energy obtained by annealing for
        the positive and negative strength sequences,
        separately.

    Notes
    -------
    Uses Maslov & Sneppen rewiring model to produce a
    surrogate adjacency matrix, B, with the same
    size, density, and degree sequence as A.
    The weights are then permuted to optimize the
    match between the strength sequences of A and B
    using simulated annealing. Positive and negative weights
    and strength sequences are treated separately.

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

    pos_A = A.copy()
    pos_A[pos_A < 0] = 0
    neg_A = A.copy()
    neg_A[neg_A > 0] = 0
    pos_s = np.sum(pos_A, axis = 1) #positive strengths of A
    neg_s = np.sum(neg_A, axis = 1) #negative strengths of A
    strengths = {'pos': pos_s, 'neg': neg_s}

    #Maslov & Sneppen rewiring
    if R is None:
        B = bct.randmio_und_signed(A, rewiring_iter, seed=seed)[0]
    else:
        B = R.copy()

    pos_B = B.copy()
    pos_B[pos_B < 0] = 0
    neg_B = B.copy()
    neg_B[neg_B > 0] = 0
    signed_B = {'pos': pos_B, 'neg': neg_B}

    B = np.zeros((n, n))
    energymin_dict = {}
    init_temp = temp
    #iteratively permuting positive and negative weights
    #to match the respective strength sequences
    for sign in ['pos', 'neg']:

        temp = init_temp

        curr_B = signed_B[sign]
        s = strengths[sign]

        u, v = np.triu(curr_B, k = 1).nonzero() #upper triangle indices
        wts = np.triu(curr_B, k = 1)[(u, v)] #upper triangle values
        m = len(wts)
        sb = np.sum(curr_B, axis = 1) #strengths of B

        if energy_func is not None:
            energy = energy_func(s, sb)
        elif energy_type == 'sse':
            energy = np.sum((s - sb)**2)
        elif energy_type == 'max':
            energy = np.max(np.abs(s - sb))
        elif energy_type == 'mae':
            energy = np.mean(np.abs(s - sb))
        elif energy_type == 'mse':
            energy = np.mean((s - sb)**2)
        elif energy_type == 'rmse':
            energy = np.sqrt(np.mean((s - sb)**2))
        else:
            msg = ("energy_type must be one of 'sse', 'max', "
                "'mae', 'mse', or 'rmse'. Received: {}.".format(energy_type))
            raise ValueError(msg)

        energymin = energy
        wtsmin = wts.copy()

        if verbose:
            print('\ninitial energy {:.5f}'.format(energy))

        for istage in tqdm(range(nstage), desc = 'annealing progress'):

            naccept = 0
            for i in range(niter):

                #permutation
                e1 = rs.randint(m)
                e2 = rs.randint(m)

                a, b = u[e1], v[e1]
                c, d = u[e2], v[e2]

                sb_prime = sb.copy()
                sb_prime[[a, b]] = sb_prime[[a, b]] - wts[e1] + wts[e2]
                sb_prime[[c, d]] = sb_prime[[c, d]] + wts[e1] - wts[e2]

                if energy_func is not None:
                    energy_prime = energy_func(sb_prime, s)
                elif energy_type == 'sse':
                    energy_prime = np.sum((sb_prime - s)**2)
                elif energy_type == 'max':
                    energy_prime = np.max(np.abs(sb_prime - s))
                elif energy_type == 'mae':
                    energy_prime = np.mean(np.abs(sb_prime - s))
                elif energy_type == 'mse':
                    energy_prime = np.mean((sb_prime - s)**2)
                elif energy_type == 'rmse':
                    energy_prime = np.sqrt(np.mean((sb_prime - s)**2))
                else:
                    msg = ("energy_type must be one of 'sse', 'max', "
                        "'mae', 'mse', or 'rmse'. "
                        "Received: {}.".format(energy_type))
                    raise ValueError(msg)

                #permutation acceptance criterion
                if (energy_prime < energy or
                    rs.rand() < np.exp(-(energy_prime - energy)/temp)):
                    sb = sb_prime.copy()
                    wts[[e1, e2]] = wts[[e2, e1]]
                    energy = energy_prime
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

        B[(u, v)] = wtsmin
        energymin_dict[sign] = energymin
    B = B + B.T

    return B, energymin


def degree_preserving_rand_sm_signed(A, rewiring_iter = 2,
                                frac = 0.5,
                                R = None, connected = None,
                                seed = None):
    try:
        A = np.asarray(A)
    except TypeError as err:
        msg = ('A must be array_like. Received: {}.'.format(type(A)))
        raise TypeError(msg) from err

    if frac > 1 or frac <= 0:
        msg = ('frac must be between 0 and 1. '
               'Received: {}.'.format(frac))
        raise ValueError(msg)

    # Maslov & Sneppen rewiring
    if R is None:
        B = bct.randmio_und_signed(A, rewiring_iter, seed=seed)[0]
    else:
        B = R.copy()
    return B

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


def rewire_signed_new(df, score_name, seed, method='SM'):
    '''keeping the node degree intact, rewire the edges. We are shuffling pairs of edges. However, as we have to consider the score
    of the new edges, we  pair up two synergistic
    edges or two nonsynergistic edges when we shuffle. '''
    edge_types = set(df['edge_type'].unique())

    rewired_df = pd.DataFrame()
    stat_df =  pd.DataFrame()
    for edge_type in edge_types:
        df_edge = df[df['edge_type'] == edge_type][['source', 'target', score_name]]
        def randomize(df_edge):
            A, node_2_idx = dataframe_to_numpy(df_edge, score_name)

            if method == 'SA': #simmulated annealing
                B, _ = strength_preserving_rand_sa_signed(A, seed=seed)
            elif method =='SM': #snepen-maslov method
                B = degree_preserving_rand_sm_signed(A, seed=seed)

            rewired_df_edge = numpy_to_dataframe(B, node_2_idx, score_name)
            rewired_df_edge['edge_type'] = edge_type
            return rewired_df_edge

        if not df_edge.empty:
            rewired_df_edge = randomize(df_edge)
        rewired_df = pd.concat([rewired_df, rewired_df_edge], axis=0)

        #check how different are the rewired and orig network
        print(f'\n\nCell line: {edge_type}')

        print('\n\nnegative edges')
        stat_df_edge = check_diff(df_edge, rewired_df_edge, score_name)
        stat_df_edge['edge_type'] = edge_type
        stat_df_edge['sign'] = 'negative'
        stat_df = pd.concat([stat_df, stat_df_edge], axis=0)

    return rewired_df, stat_df

def rewire_unsigned(df, score_name, seed, method='SA'):
    '''keeping the node degree intact, rewire the edges. We are shuffling pairs of edges. However, as we have to consider the score
    of the new edges, we  pair up two synergistic
    edges or two nonsynergistic edges when we shuffle. '''
    print(df.head(5))
    edge_types = set(df['edge_type'].unique())

    rewired_df = pd.DataFrame()
    stat_df =  pd.DataFrame()
    for edge_type in edge_types:
        df_edge_pos = df[(df['edge_type'] == edge_type)&(df[score_name]>=0)][['source', 'target', score_name]]
        df_edge_neg =  df[(df['edge_type'] == edge_type)&(df[score_name]<0)][['source', 'target', score_name]]
        def randomize(df_edge):
            A, node_2_idx = dataframe_to_numpy(df_edge, score_name)

            if method == 'SA': #simmulated annealing
                B, _ = strength_preserving_rand_sa(A, seed=seed)
            elif method =='SM': #snepen-maslov method
                B = degree_preserving_rand_sm(A, seed=seed)
                # B = custom_preserving_rand_sm_weighted(A, seed=seed)

            rewired_df_edge = numpy_to_dataframe(B, node_2_idx, score_name)
            rewired_df_edge['edge_type'] = edge_type
            return rewired_df_edge

        rewired_df_pos = pd.DataFrame()
        rewired_df_neg = pd.DataFrame()
        if not df_edge_pos.empty:
            rewired_df_pos= randomize(df_edge_pos)
        if not df_edge_neg.empty:
            rewired_df_neg = randomize(df_edge_neg)
        rewired_df = pd.concat([rewired_df, rewired_df_pos, rewired_df_neg], axis=0)

        #check how different are the rewired and orig network
        print(f'\n\nCell line: {edge_type}')
        print('positive edges')
        df_edge_pos['edge_type']=edge_type
        df_edge_neg['edge_type']=edge_type
        pos_stat_df = check_diff(df_edge_pos, rewired_df_pos, score_name)
        pos_stat_df['edge_type']=edge_type
        pos_stat_df['sign'] = 'positive'
        print('\n\nnegative edges')
        neg_stat_df = check_diff(df_edge_neg, rewired_df_neg, score_name)
        neg_stat_df['edge_type'] = edge_type
        neg_stat_df['sign'] = 'negative'
        stat_df = pd.concat([stat_df, pos_stat_df, neg_stat_df], axis=0)

    rewired_df = rewired_df.sample(frac=1)
    return rewired_df, stat_df




def get_rewired_train_val (all_train_df, score_name, method, split_type, val_frac, seed, rewired_train_file, force_run=False):
    if (not os.path.exists(rewired_train_file))|force_run:
        os.makedirs(os.path.dirname(rewired_train_file), exist_ok=True)
        rewired_train_df, stat_df = rewire_unsigned(all_train_df, score_name, seed, method=method)
        stat_df.to_csv(rewired_train_file.replace('.tsv', '_stat.tsv'), sep='\t', index=False)
        # sort the df so that if the rewired network are same for the same seed, it will be saved as the same. and then
        # while splitting into train-val, we will get the same split.
        sort_cols = ['source', 'target', 'edge_type', score_name]
        rewired_train_df = rewired_train_df.sort_values(by=sort_cols).reset_index(drop=True)
        #save the rewired network
        rewired_train_df.to_csv(rewired_train_file, sep='\t', index=False)

    else:
        print('Loading rewired network')
        rewired_train_df = pd.read_csv(rewired_train_file, sep='\t')

    #make sure the index are integer values.
    rewired_train_df[['source', 'target', 'edge_type']] = rewired_train_df[
        ['source', 'target', 'edge_type']].astype(int)

    # sort rewired_train_df so that (a,b) and (b,a) edges appear as (max(a,b), min(a,b))
    sort_paired_cols(rewired_train_df, 'source', 'target', inplace=True, relation='greater')

    split_type_map = {'random': 'random', 'leave_comb': 'edge', 'leave_drug': 'node', 'leave_cell_line': 'edge_type'}
    train_idx, val_idx = split_train_test(rewired_train_df, split_type_map[split_type], val_frac, seed=0)

    check_diff(all_train_df, rewired_train_df, score_name)


    edge_types = set(all_train_df['edge_type'].unique())
    for edge_type in edge_types:
        df1=all_train_df[all_train_df['edge_type'] == edge_type]
        df2=rewired_train_df[rewired_train_df['edge_type'] == edge_type]
        orig_nodes = set(df1['source']).union(set(df1['target']))
        rewiered_nodes = set(df2['source']).union(set(df2['target']))
        print('uncommon: ', orig_nodes.difference(rewiered_nodes))
        print('total original vs rewired edge:', len(df1), len(df2))
        removed_edgeweight = set(df1[score_name]).difference(set(df2[score_name]))
        for e in removed_edgeweight:
            print(df1[df1[score_name]==e])




    return rewired_train_df, {0:train_idx}, {0:val_idx}

def check_diff(all_train_df, unsorted_rewired_train_df, score_name ):
    # Combine 'source' and 'target' columns to count participation in edges
    all_nodes = pd.concat([all_train_df['source'], all_train_df['target']], ignore_index=True)

    # Count occurrences
    node_counts = all_nodes.value_counts().reset_index()
    node_counts.columns = ['node', 'count']

    # Optional: sort by count
    node_counts = node_counts.sort_values(by='count', ascending=False)

    # Display or inspect
    # print(node_counts.head())

    rewired_train_df = sort_paired_cols(unsorted_rewired_train_df, 'source', 'target', inplace=False, relation='greater')

    orig_triplets = set(zip(all_train_df['source'], all_train_df['target'], all_train_df['edge_type']))
    rewired_triplet = set(zip(rewired_train_df['source'], rewired_train_df['target'], rewired_train_df['edge_type']))
    common_triplets = orig_triplets.intersection(rewired_triplet)

    orig_quad = set(zip(all_train_df['source'], all_train_df['target'], all_train_df['edge_type'], all_train_df[score_name]))
    rewired_quad = set(zip(rewired_train_df['source'], rewired_train_df['target'], rewired_train_df['edge_type'], all_train_df[score_name]))
    common_quad = orig_quad.intersection(rewired_quad)

    print(f'total samples in original {len(orig_quad)} vs. in rewired {len(rewired_quad)}')
    print(f'total triplets in original {len(orig_triplets)} vs. in rewired {len(rewired_triplet)}')

    # print(f'frac common triplets: {len(common_triplets)/len(total_triplets)} among total of: {len(total_triplets)}')
    print(f'frac triplet from original: {len(common_triplets)/len(rewired_triplet)} among total of: {len(rewired_triplet)}')

    # print(f'frac common quads: {len(common_quad)/len(total_quad)} among total of: {len(total_quad)}')
    print(f'frac samples from original: {len(common_quad)/len(rewired_quad)} among total of: {len(rewired_quad)}')

    merged = compute_deviation_of_score(all_train_df, rewired_train_df, score_name)

    # Step 4: Unique nodes and edge statistics (undirected)
    unique_nodes = set(all_train_df['source']).union(set(all_train_df['target']))
    n_nodes = len(unique_nodes)
    n_possible_edges = n_nodes * (n_nodes - 1) // 2

    # Drop duplicates for undirected edges
    undirected_edges = pd.DataFrame(np.sort(all_train_df[['source', 'target']].values, axis=1),
                                    columns=['node1', 'node2'])
    n_present_edges = len(undirected_edges.drop_duplicates())

    # Step 5: Store all computed statistics in a DataFrame
    stats = {
        'n_orig_quad': [len(orig_quad)],
        'n_rewired_quad': [len(rewired_quad)],
        'n_orig_triplets': [len(orig_triplets)],
        'n_rewired_triplets': [len(rewired_triplet)],
        'n_common_triplets': [len(common_triplets)],
        'frac_triplets_from_original': [len(common_triplets) / len(rewired_triplet) if rewired_triplet else np.nan],
        'n_common_quad': [len(common_quad)],
        'frac_samples_from_original': [len(common_quad) / len(rewired_quad) if rewired_quad else np.nan],
        'avg_score_diff': [merged['score_diff'].mean()],
        'n_unique_nodes': [n_nodes],
        'n_possible_edges': [n_possible_edges],
        'n_present_edges': [n_present_edges],
        'frac_edges_present': [n_present_edges / n_possible_edges if n_possible_edges else np.nan],
    }

    df_stats = pd.DataFrame(stats)
    return  df_stats

def compute_deviation_of_score(all_train_df , rewired_train_df, score_name ):
    # find deviation of score among the overlapping triplets between original and rewired network
    merged = all_train_df[['source', 'target', 'edge_type', score_name]] \
        .merge(
        rewired_train_df[['source', 'target', 'edge_type', score_name]],
        on=['source', 'target', 'edge_type'],
        suffixes=('_orig', '_rewired')
    )
    # 2. Compute the difference (orig minus rewired)
    merged['score_diff'] = abs(merged[f'{score_name}_orig'] - merged[f'{score_name}_rewired'])
    print(f'average difference btn the same triplet present in original and newired network: ',
          merged['score_diff'].mean())
    return merged
def check_dups(rewired_train_df):
    # check for edges like (a,b) and (b, a)
    rev_pairs = set(zip(rewired_train_df['source'], rewired_train_df['target'])).intersection(
        set(zip(rewired_train_df['target'], rewired_train_df['source'])))
    print(f'#duplicated pairs in rewired: {len(rev_pairs)}')

    dup_triplets = rewired_train_df.duplicated(subset=['source', 'target', 'edge_type']).sum()
    print(f'#duplicated triplets in rewired: {dup_triplets}')
