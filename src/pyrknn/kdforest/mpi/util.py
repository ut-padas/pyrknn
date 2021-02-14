import numpy as np
import numba
import os

from ...kernels.cpu import core as cpu

if os.environ["PYRKNN_USE_CUDA"] == '1':
    from ...kernels.gpu import core as gpu
    from ...kernels.gpu import core_sparse as gpu_sparse
    from numba import cuda
    import cupy as cp
else:
    import numpy as cp
    from ...kernels.cpu import core as gpu
    from ...kernels.cpu import core as gpu_sparse

import time

"""File that contains key kernels to be replaced with high performance implementations"""

class Recorder:
    record_book = dict()
    
    def push(self, string, value):
        if string in self.record_book:
            self.record_book[string].append(value)
        else:
            self.record_book[string] = [value]

    def print(self, string=None):
        if string == None:
            print("Record Book")
            for n, t in self.record_book.items():
                print('{} {}'.format(n, t))
        elif string in self.record_book:
            print(self.record_book[string])

    def write(self, filename):
        with open(filename, 'w') as f:
            for key in self.record_book.keys():
                f.write("%s"%(key))
                for item in self.record_book[key]:
                    f.write(",%s"%(item))
                f.write("\n")


class Profiler:

    output_times  = dict()
    current_times = dict()

    def push(self, string):
        if string in self.current_times and self.current_times[string] != 0:
            print("Warning: Resetting currently running timer")
        self.current_times[string] = time.time()

    def pop(self, string):
        if string not in self.current_times:
            raise Exception("Error: Trying to pop Time Region that does not exist")
        if string in self.current_times and self.current_times[string] == 0:
            raise Exception("Error: Trying to pop Time Region that is not running")

        if string in self.output_times:
            self.output_times[string] += ( time.time() - self.current_times[string] )
        else:
            self.output_times[string] = ( time.time() - self.current_times[string] )

        #Reset
        self.current_times[string] = 0

    def reset(self, string=None):
        if string is not None:
            self.current_times[string] = 0
            self.output_times[string] = 0
        else:
            self.current_times = dict()
            self.output_times  = dict()

    def print(self):
        print("Region Time(s)")
        for n, t in self.output_times.items():
            print('{} {}'.format(n, t))

    def write(self, filename):
        with open(filename, 'w') as f:
            for key in self.output_times.keys():
                f.write("%s,%s\n"%(key,self.output_times[key]))


 

#Note: Data movement should be done on the python layer BEFORE this one
#      All functions can assume all data is local to their device (should this be asserted and checked here? for debugging yes)

#The Large TODO list
#TODO: 1 Add all C++ functions from Hongru (that wrap kernels from Chenhan's GOFMM and Bo Xiao's RKDT)
#TODO: 2 Add all CUDA function from Hongru and my own
#TODO: 3 Add switching mechanism based on: Task Context? Where the data is? etc...

accuracy = []
diff=[]

env = "CPU" #Coarse kernel context switching mechanism for debugging (this is a bad pattern, to be replace in #TODO 3)

lib = np
batched = 1
sparse = 0

cores = 0

def set_env(loc, sp):
    global env
    env = loc

    global sparse
    sparse = sp

    global lib

    if env == "CPU" or env == "PYTHON":
        lib = np
    elif env == "GPU":
        lib = cp
    else:
        raise Exception("Not a valid enviorment target. Specify CPU or GPU")

def set_cores(c):
    global cores
    cores = c

device = 0
def set_device(c):
    global device
    device = c


#TODO: Q2 and R2 could be precomputed and given as arguments to distance(), should make keyword parameters for this option
def distance(R, Q):
    """Compute the distances between a reference set R (|R| x d) and query set Q (|Q| x d).

    Result:
        D -- |Q| x |R| resulting distance matrix.
        Note: Positivity not enforced for rounding errors in distances near zero.
    """
    global env
    if env == "PYTHON" or env == "CPU" or env=="GPU":
        D= -2*lib.dot(Q, R.T)                    #Compute -2*<q_i, r_j>, Note: Python is row-major
        Q2 = lib.linalg.norm(Q, axis=1)**2       #Compute ||q_i||^2
        R2 = lib.linalg.norm(R, axis=1)**2       #Compute ||r_j||^2

        D = D + Q2[:, np.newaxis]               #Add in ||q_i||^2 row-wise
        D = D + R2                              #Add in ||q_i||^2 colum-wise
        return D


    """
    if env == "CPU":
        return cpu.distance(R, Q);
    """

def single_knn(gids, R, Q, k):
    """Compute the k nearest neighbors from a query set Q (|Q| x d) in a reference set R (|R| x d).

    Note: Neighbors are indexed globally by gids, which provides a local (row idx) to global map.

    Return:
        (neighbor_list , neigbor_dist)
        neighbor_list -- |Q| x k list of neighbor gids. In local ordering og the query points.
        neighbor_dist -- the corresponding distances

    """

    global env
    global sparse
    global cores

    if env == "PYTHON":
        #Note: My Pure python implementation is quite wasteful with memory.

        N, d, = Q.shape

        #Allocate storage space for neighbor ids and distances
        neighbor_list = lib.zeros([N, k])                #TODO: Change type to int
        neighbor_dist = lib.zeros([N, k])

        dist = distance(R, Q)                           #Compute all distances

        for q_idx in range(N):

            #TODO: Replace with kselect kernel
            #TODO: neighbor_dist, neighbor_list  = np.kselect(dist, k, key=gids)

            #Perform k selection on distance list
            neighbor_lids = lib.argpartition(dist[q_idx, ...], k)[:k]            #This performs a copy of dist and allocated lids
            neighbor_dist[q_idx, ...] = dist[q_idx, neighbor_lids]              #This performs a copy of dist
            neighbor_gids = gids[neighbor_lids]                                 #This performs a copy of gids

            #Sort and store the selected k neighbors
            shuffle_idx = lib.argsort(neighbor_dist[q_idx, ...])                 #This performs a copy of dist and allocates idx
            neighbor_dist[q_idx, ...] = neighbor_dist[q_idx, shuffle_idx]       #This performs a copy of dist
            neighbor_gids = neighbor_gids[shuffle_idx]                          #This performs a copy of gids

            neighbor_list[q_idx, ...] = neighbor_gids                           #This performs a copy of gids

        return np.asarray(neighbor_list, dtype=np.int32), neighbor_dist

    elif not sparse:
        return cpu.single_knn(gids, R, Q, k, cores)
    else:
        print("Running Sparse Exact")
        return cpu.sparse_exact(gids, R, Q, k, cores)

def batched_knn(gidsList, RList, QList, k):
   if env == "CPU":
        return cpu.batched_knn(gidsList, RList, QList, k, cores)
   if env == "GPU":
        raise Exception("Error: GPU Batched KNN no longer supported through this interface. Use combined kernel `dense_knn` instead.")
        #return gpu.batched_knn(gidsList, RList, QList, k)

def multileaf_knn(gidsList, RList, QList, k):

    if batched and env != "PYTHON":
        return batched_knn(gidsList, RList, QList, k)
    else:
        N = len(gidsList)

        NLL = []
        NDL = []

        for i in range(N):
            NL, ND  = single_knn(gidsList[i], RList[i], QList[i], k)
            NLL.append(NL)
            NDL.append(ND)

        return (NLL, NDL, None)

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
    if env == "PYTHON" or env=="GPU":
        #A pure python implementation
        lids = lib.argpartition(values, k)[:k]
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
    merge_t = time.time()

    if env == "CPU":
        return cpu.merge_neighbors(a, b, k, cores)

    if env == "GPU":
        out = gpu.merge_neighbors(a, b, k, device)

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

        out = (np.asarray(a_list, dtype=np.int32), a_dist)

    merge_t = time.time() - merge_t

    return out

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

    N = Na
    k = ka

    knndist = a_dist[:, k-1]
    err = 0
    relative_distance = 0

    for i in range(Na):
        miss_array_id   = [0 if a_list[i, j] in b_list[i] else 1 for j in range(k)]
        miss_array_dist = [0 if b_dist[i, j] < knndist[i] else 1 for j in range(k)]
        miss_array = np.logical_or(miss_array_id, miss_array_dist)
        #miss_array = miss_array_id
        err+= np.sum(miss_array)
        relative_distance = max(relative_distance, np.abs(knndist[i] - b_dist[i, kb-1])/knndist[i])

    perc = 1 - float(err)/(Na*ka)

    return perc, relative_distance

def dist_select(k, data, ids, comm):
    return cpu.dist_select(k, data, ids, comm)

def cpu_sparse_knn(gids, X, levels, ntrees, k, blocksize):
    return cpu.sparse_knn(gids, X, levels, ntrees, k, blocksize, cores)

def gpu_sparse_knn(gids, X, levels, ntrees, k, blockleaf, blocksize, device):
    return gpu_sparse.sparse_knn(gids, X, levels, ntrees, k, blockleaf, blocksize, device)

def dense_knn(gids, X, levels, ntrees, k, blocksize, device):
    return gpu.dense_knn(gids, X, levels, ntrees, k, blocksize, device)

def neighbor_order(neighbor_list, neighbor_dist, NLL, NDL, size, k):
    print("NOT COMPLETE")


