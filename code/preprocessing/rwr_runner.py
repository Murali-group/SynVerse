import pandas as pd
import os
import pickle
from scipy.sparse import csr_matrix, csc_matrix, find
import numpy as np

def prepare_network(net_file, prot_info_file, drug_target_df, confidence_threshold=900, force_run=False):

    W_out_file = f'{os.path.dirname(net_file)}/W_{confidence_threshold}.pickle'
    gene_idx_out_file = f'{os.path.dirname(net_file)}/gene_to_idx_{confidence_threshold}.pickle'

    if (os.path.exists(W_out_file) and os.path.exists(gene_idx_out_file) and (force_run==False)):
        with open(W_out_file, 'rb') as f:
            sparse_adj_matrix = pickle.load(f)
        with open(gene_idx_out_file, 'rb') as f:
            name_to_index = pickle.load(f)
        print('done loading network')


    else:
        #parse network
        net_df = pd.read_csv(net_file, sep=' ', compression='gzip')
        prot_info_df = pd.read_csv(prot_info_file, sep='\t', compression='gzip')

        merged_df = net_df.merge(prot_info_df[['#string_protein_id', 'preferred_name']],
                                 left_on='protein1',
                                 right_on='#string_protein_id',
                                 how='left').rename(columns={'preferred_name': 'source'})

        merged_df = merged_df.merge(prot_info_df[['#string_protein_id', 'preferred_name']],
                                    left_on='protein2',
                                    right_on='#string_protein_id',
                                    how='left').rename(columns={'preferred_name': 'target'})

        # Select only the required columns
        string_df = merged_df[['source', 'target', 'combined_score']]

        #filter out the low confidence edges
        string_df = string_df[string_df['combined_score']>confidence_threshold]
        genes = set(string_df['source']).union(string_df['target'])
        print(f'unique genes in STRING: {len(genes)}')
        print(f'edges in STRING: {len(string_df)}')
        # print(result_df.head(5))

        #convert pandas frame into numpy matrix
        string_genes = pd.unique(string_df[['source', 'target']].values.ravel('K'))

        #incporporate genes absent in STRING but present as drug target to the network
        target_genes = list(drug_target_df.columns)

        uncommon = set(target_genes).difference(set(string_genes))
        all_genes = list(set(target_genes).union(set(string_genes)))

        #create the adjacency matrix
        all_genes.sort()
        # Create a mapping from preferred names to indices
        name_to_index = {name: idx for idx, name in enumerate(all_genes)}

        # Initialize an empty adjacency matrix
        adj_matrix = np.zeros((len(all_genes), len(all_genes)), dtype=int)

        # Populate the adjacency matrix
        for _, row in string_df.iterrows():
            i = name_to_index[row['source']]
            j = name_to_index[row['target']]
            adj_matrix[i, j] = 1
            adj_matrix[j, i] = 1  # Since the graph is undirected
        print('done preparing network')

        sparse_adj_matrix = csr_matrix(adj_matrix)

        with open(W_out_file, 'wb') as f:
            pickle.dump(sparse_adj_matrix, f)
        with open(gene_idx_out_file, 'wb') as f:
            pickle.dump(name_to_index, f)

    print(f'STRING with threshold {confidence_threshold}: #nodes {len(sparse_adj_matrix.shape[0])}, '
          f'#edges: {sparse_adj_matrix.nnz}')
    return sparse_adj_matrix, name_to_index


def create_transition_mtx_zerodegnodes(net, N, alpha):
    '''
    The following function will create a transition matrix considering
    targets along the rows and sources along the columns.
    '''
    outDeg = csr_matrix((1, N), dtype=float)
    zeroDegNodes = set()
    for i in range(N):
        #NURE: till 12/30 the following was implemented. But the net was default
        # symmetric till this point.
        # outDeg[i, 0] = 1.0 * net.getrow(i).sum()  # weighted out degree of every node
        #NURE: from 12/30. As for directed network we have source of an edge along columns,
        # for computing outdegree we need to sum column wise.
        outDeg[0, i] = 1.0 * net.getcol(i).sum()  # weighted out degree of every node

        if outDeg[0, i] == 0:
            zeroDegNodes.add(i)
    # Walking in from neighbouring edge:
    #Nure: Checked throughly the following statement to make sure that the degree normalization is done column wise.
    e = net.multiply(1 - alpha).multiply(outDeg.power(-1))  # (1-q)*net[u,v]/(outDeg[u])
    return e, zeroDegNodes
def rwr(net, weights={}, alpha=0.5, eps=0.01, maxIters=500, verbose=False, weightName='weight'):
    N = net.get_shape()[0]
    # print("Shape of network = ", net.get_shape())
    # print("Number of teleportation weights = ", len(weights))
    # print("Number of nodes in network = ", N)

    ###### Create transition matrix ###################
    X, zeroDegNodes = create_transition_mtx_zerodegnodes(net, N, alpha)
    incomingTeleProb = {}  # The node weights when the algorithm begins, also used as teleport-to probabilities

    # Find the incoming teleportation probability for each node, which is also used as the initial probabilities in
    # the graph. If no node weights are passed in, use a uniform distribution.
    totalWeight = sum([w for v, w in weights.items()])

    # If weights are given, apply two transformations
    #   - Add a small incoming teleportation probability to every node to ensure that the graph is strongly connected
    #   - Normalize the weights to sum to one: these are now probabilities.
    # Find the smallest non-zero weight in the graph.

    minPosWeight = 1.0
    for v, weight in weights.items():
        if weight == 0:
            continue
        minPosWeight = min(minPosWeight, 1.0 * weight / totalWeight)

    # The epsilon used as the added incoming teleportation probabilty is 1/1000000 the size of the smallest weight given
    # so that it does not impact results.
    smallWeight = minPosWeight / (10 ** 6)

    for i in range(N):
        weight = weights.get(i, 0.0)
        incomingTeleProb[i] = 1.0 * (weight + smallWeight) / (totalWeight + smallWeight * N)

    # Sparse matrices to store the probability scores of the nodes
    currVisitProb = csr_matrix([list(incomingTeleProb.values())], dtype=float)  # currVisitProb: 1 X N
    # prevVisitProb must converted to N X 1 to multiply with Transition Matrix (N X N) and yield new currVisitProb(N X 1)
    prevVisitProb = currVisitProb.transpose()  # prevVisitProb: N X 1


    # Teleporting from source node:
    # currVisitProb holds values of incomingTeleProb
    t = currVisitProb.multiply(alpha).transpose()  # (q)*(incomingTeleProb[v]
    # print("Shape of matrix t = ", t.shape)  # N X 1

    iters = 0
    finished = False

    while not finished:
        iters += 1

        # X: N X N ; prevVisitProb: N X 1 ; Thus, X.transpose() * prevVisitProb: N X 1
        #Nure: till 12/30, in X along rows I had sources, and columns I had targets.
        # currVisitProb = (X.transpose() * prevVisitProb)

        #Nure: from 12/30, in X along rows I have sources, and columns I have targets. So using
        # X directly instead of X.transpose()
        currVisitProb = (X * prevVisitProb)
        currVisitProb = currVisitProb + t  # N X 1


        # Teleporting from dangling node
        # In the basic formulation, nodes with degree zero ("dangling
        # nodes") have no weight to send a random walker if it does not
        # teleport. We consider a walker on one of these nodes to
        # teleport with uniform probability to any node. Here we compute
        # the probability that a walker will teleport to each node by
        # this process.
        zSum = sum([prevVisitProb[x, 0] for x in zeroDegNodes])/N  # scalar
        currVisitProb = currVisitProb + csc_matrix(np.full(currVisitProb.shape,
                        ((1 - alpha) * zSum)))  # the scalar (1-q)*zSum will get broadcasted and added to every element of currVisitProb

        # currVisitProb = currVisitProb + ((1 - alpha) * zSum)  # the scalar (1-q)*zSum will get broadcasted and added to every element of currVisitProb

        # Keep track of the maximum RELATIVE difference between this
        # iteration and the previous to test for convergence
        maxDiff = (abs(prevVisitProb - currVisitProb) / currVisitProb).max()

        # Print statistics on the iteration
        # print("\tIteration %d, max difference %f" % (iters, maxDiff))
        if maxDiff < eps:
            print("RWR converged after %d iterations, max difference %f" % (iters, maxDiff))

        # Give a warning if termination happens by the iteration cap,
        # which generally should not be expected.
        if iters >= maxIters:
            print("WARNING: RWR terminated because max iterations (%d) was reached." % (maxIters))

        # Test for termination, either due to convergence or exceeding the iteration cap
        finished = (maxDiff < eps) or (iters >= maxIters)
        # Update prevVistProb
        prevVisitProb = currVisitProb


        # break
    # Create a dictionary of final scores (keeping it consistent with the return type in PageRank.py)
    finalScores = {}
    for i in range(N):
        finalScores[i] = currVisitProb[i, 0]  #dim(currVisitProb)=N*1.
        # so, take the value of the first col    which is the only col as well
    return finalScores


def rwr_wrapper(drug_target_df, net_file=None, prot_info_file=None, alpha=0.5, out_file=None, force_run=True):
    '''
    :param W: PPI network
    :param positives: the seed nodes
    :param alpha: restart parameter
    :return:
    '''

    variance_threshold = 0.0000001
    if (os.path.exists(out_file) and not force_run):
        rwr_score_df = pd.read_csv(out_file,dtype={'pid':str}, sep='\t', index_col=None)
    else:
        drug_target_df.set_index('pid', inplace=True)
        W, gene_to_idx = prepare_network(net_file, prot_info_file, drug_target_df,
                                         confidence_threshold=900, force_run=force_run)

        rwr_score={}
        for i, row in drug_target_df.iterrows():
            positive_weights = {gene_to_idx[col]:
                                row[col] for col in drug_target_df.columns if row[col] == 1}
            rwr_score[i] = rwr(W, weights=positive_weights,
                            alpha=alpha, eps=0.01, maxIters=500,
                            verbose=False)
        drug_target_df.reset_index(inplace=True)
        rwr_score_df = pd.DataFrame.from_dict(rwr_score, orient='index').rename_axis('pid').reset_index()
        rwr_score_df.to_csv(out_file, sep='\t', index=False)

    #filter out the genes with 0 variance.
    rwr_score_df.set_index('pid', inplace=True)
    rwr_score_df = rwr_score_df.loc[:, rwr_score_df.var()>variance_threshold]
    rwr_score_df.reset_index(inplace=True)

    return rwr_score_df
