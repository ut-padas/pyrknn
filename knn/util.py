import numpy as np

"""File that contains key kernels to be replaced with high performance implementations"""

#Note: Data movement should be done on the python layer BEFORE this one
#      All functions can assume all data is local to their device (should this be asserted and checked here? for debugging yes)

#The Large TODO list
#TODO: 1 Add all C++ functions from Hongru (that wrap kernels from Chenhan's GOFMM and Bo Xiao's RKDT)
#TODO: 2 Add all CUDA function from Hongru and my own
#TODO: 3 Add switching mechanism based on: Task Context? Where the data is? etc...


env = "PYTHON" #Coarse kernel context switching mechanism for debugging (this is a bad pattern, to be replace in #TODO 3)
#Options are: PYTHON, CPU, GPU


#TODO: Q2 and R2 could be precomputed and given as arguments to distance(), should make keyword parameters for this option
def distance(R, Q):
    """Compute the distances between a reference set R (|R| x d) and query set Q (|Q| x d).

    Result:
        D -- |Q| x |R| resulting distance matrix.
        Note: Positivity not enforced for rounding errors in distances near zero.
    """
    global env
    if env == "PYTHON":
        D= -2*np.dot(Q, R.T)                    #Compute -2*<q_i, r_j>, Note: Python is row-major
        Q2 = np.linalg.norm(Q, axis=1)**2       #Compute ||q_i||^2
        R2 = np.linalg.norm(R, axis=1)**2       #Compute ||r_j||^2

        D = D + Q2[:, np.newaxis]               #Add in ||q_i||^2 row-wise
        D = D + R2                              #Add in ||q_i||^2 colum-wise
        return D


def direct_knn(gids, R, Q, k):
    """Compute the k nearest neighbors from a query set Q (|Q| x d) in a reference set R (|R| x d).

    Note: Neighbors are indexed globally by gids, which provides a local (row idx) to global map.

    Return:
        (neighbor_list , neigbor_dist)
        neighbor_list -- |Q| x k list of neighbor gids. In local ordering og the query points.
        neighbor_dist -- the corresponding distances

    """
    global env
    if env == "PYTHON":
        #Note: My Pure python implementation is quite wasteful with memory.

        N, d, = Q.shape

        #Allocate storage space for neighbor ids and distances
        neighbor_list = np.zeros([N, k])                #TODO: Change type to int
        neighbor_dist = np.zeros([N, k])

        dist = distance(R, Q)                           #Compute all distances
        for q_idx in range(N):

            #TODO: Replace with kselect kernel
            #TODO: neighbor_dist, neighbor_list  = np.kselect(dist, k, key=gids)

            #Perform k selection on distance list
            neighbor_lids = np.argpartition(dist[q_idx, ...], k)[:k]            #This performs a copy of dist and allocated lids
            neighbor_dist[q_idx, ...] = dist[q_idx, neighbor_lids]              #This performs a copy of dist
            neighbor_gids = gids[neighbor_lids]                                 #This performs a copy of gids

            #Sort and store the selected k neighbors
            shuffle_idx = np.argsort(neighbor_dist[q_idx, ...])                 #This performs a copy of dist and allocates idx
            neighbor_dist[q_idx, ...] = neighbor_dist[q_idx, shuffle_idx]       #This performs a copy of dist
            neighbor_gids = neighbor_gids[shuffle_idx]                          #This performs a copy of gids

            neighbor_list[q_idx, ...] = neighbor_gids                           #This performs a copy of gids

        return neighbor_list, neighbor_dist

def kselect(values, k, key=None):
    """ Select the k smallest elements rowwise from values.

    Keyword Arguments:
        key -- keys to be returned along with the k smallest elements of values

    Return:
        if key is not None: (k_smallest_values, k_smallest_keys)
        if key is None: (k_smallest_values)

        k_smallest_values -- k smallest values of 'values', not sorted
        k_smallest_keys -- keys corresponding to the values returned in the order of k_smallest_values
    """
    global env
    if env == "PYTHON":
        #A pure python implementation
        lids = np.argpartition(values, k)[:k]
        k_smallest_values = values[lids]
        k_smallest_keys = None
        if keys is not None:
           k_smallest_keys = keys[lids]

        return k_smallest_values, k_smallest_keys

#TODO: At the moment I've been using python's default dictionary to do this functionality. Should be made into a kernel I can run in parallel or on GPU.
##     Useful reference: https://dl.acm.org/doi/abs/10.1145/3108139
def multisplit(array, keys):
    """Partition the data into #unique(keys) buckets

    Note: array will be overwritten (possibly, impl doesn't exist yet....)

    Return:
        array -- partitioned array in #unique(keys)
        keys  -- partitioned keys in #unique(keys)
        buckets -- map from keys to start/stop indices in partitioned array
    """

    buckets = None #Should be start/stops indexed on key

    return array, keys, buckets

def merge_neighbors(a, b, k):
    """Merge nearest neighbor results from different trees

    Arguments:
        a, b -- nearest neighbor results of the tuple form (neighbor_list, neighbor_dist)
        k    -- number of nearest neighbors to keep

    Note/Warning: 'a' should be changed during the course of this function

    Return:
        (a_list, a_dist) -- updated a
        changes -- number of updates made to the nearest neihgbor list a
    """
    global env
    if env == "PYTHON":
        # This is currently very wasteful/suboptimal
        # Just a reference implementation

        a_list = a[0]
        a_dist = a[1]

        b_list = b[0]
        b_dist = b[1]

        Na, ka = a_list.shape
        Nb, kb = b_list.shape
        assert(Na == Nb)
        assert(ka+kb > k)

        changes = 0
        for i in range(Na):
            merged_dist = np.concatenate((a_dist[i], b_dist[i]))
            merged_idx  = np.concatenate((a_list[i], b_list[i]))

            #filter unique elements
            merged_idx, unique_set = np.unique(merged_idx, return_index=True)
            merged_dist = merged_dist[unique_set]

            if len(merged_idx) > k:

                #kselect
                idx = np.argpartition(merged_dist, k)[:k]
                merged_idx = merged_idx[idx]
                merged_dist = merged_dist[idx]

            #sort
            sorted_set = np.argsort(merged_dist)
            merged_dist = merged_dist[sorted_set]
            merged_idx = merged_idx[sorted_set]

            #count changes (#TODO: Is this a useful statistic to keep, should it be part of the kernel?)
            changes += (k - np.sum(merged_idx == a_list[i]))

            #store
            a_list[i] = merged_idx
            a_dist[i] = merged_dist

        return (a_list, a_dist), changes

#This isn't really an HPC kernel, just a useful utility function I didn't know where else to put
def neighbor_dist(a, b):
    """Compute how accurate the nearest neighbors are.

    Arguments:
        a, b -- nearest neighbor sets
        a -- assumed truth (for relative error calculation)

    Return:
        perc -- percent of incorrect nearest neighbors (0.0 is perfect recall accuracy)
        nn_dist -- mean of the relative error in the kth nearest neighbor distance
        first_diff -- |Q| length array, each entry is the location of the first incorrect nearest neighbor i. Results are perfect recall < ith entry.

    """
    a_list = a[0]
    a_dist = a[1]

    b_list = b[0]
    b_dist = b[1]

    Na, ka = a_list.shape
    Nb, kb = b_list.shape
    assert(Na == Nb)
    assert(ka == kb)

    changes = 0
    nn_dist = np.zeros(Na)
    first_diff = np.zeros(Na)
    for i in range(Na):
        changes += (ka - np.sum(a_list[i] == b_list[i]))
        nn_dist[i] = np.abs(a_dist[i, -1] - b_dist[i, -1])/np.abs(a_dist[i, -1])
        diff_array = np.abs(a_list[i, ...] - b_list[i, ...])
        first_diff[i] = np.argmax(diff_array > 0.5)

    perc = changes/(Na*ka)

    return perc, np.mean(nn_dist), first_diff







