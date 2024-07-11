from scipy.sparse import csr_matrix, csc_matrix, find
import numpy as np
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

    # TODO: handle no incoming weights

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
