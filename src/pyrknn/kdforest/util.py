
from .config import *

import numpy as np
import numba
import os

import scipy as sp
from scipy import sparse as sparse

from ..kernels.cpu import core as cpu

use_cuda = PYRKNN_USE_CUDA
numba_threads = PYRKNN_NUMBA_THREADS

if use_cuda:
    from ..kernels.gpu import core_gpu as gpu
    from ..kernels.gpu import core_sparse as gpu_sparse
    from numba import cuda
    import cupy as cp
else:
    import numpy as cp
    from ..kernels.cpu import core as gpu
    from ..kernels.cpu import core as gpu_sparse

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
        #print(string)
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
            print('{}, {}'.format(n, t))

    def write(self, filename):
        with open(filename, 'w') as f:
            for key in self.output_times.keys():
                f.write("%s,%s\n"%(key,self.output_times[key]))


 

#Note: Data movement should be done on the python layer BEFORE this one
#      All functions can assume all data is local to their device (should this be asserted and checked here? for debugging yes)

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

def direct_knn(ids, R, Q, k, loc="HOST", cores=8):

    """Compute the k nearest neighbors from a query set Q (|Q| x d) in a reference set R (|R| x d).

    Note: Neighbors are indexed globally by gids, which provides a local (row idx) to global map.

    Return:
        (neighbor_list , neigbor_dist)
        neighbor_list -- |Q| x k list of neighbor gids. In local ordering og the query points.
        neighbor_dist -- the corresponding distances

    """

    sparse_flag = isinstance(R, sp.sparse.csr.csr_matrix) and isinstance(Q, sp.sparse.csr.csr_matrix)
    dense_flag = isinstance(R, np.ndarray) and isinstance(Q, np.ndarray)

    #TODO: Currently we only support these direct searches on the CPU
    #TODO: Gather these and isolate these kernels from GPU code
 
    assert(loc == "HOST")
 
    if dense_flag:
        return cpu.single_knn(ids, R, Q, k, cores)
    elif sparse_flag:
        #print("Running Sparse Exact")
        return cpu.sparse_exact(ids, R, Q, k, cores)
    
def batched_knn(ridsList, RList, QList, k, qidsList=None, neighbor_ids=None, neighbor_dist=None, n=None, gids=None, repack=True):
   if env == "CPU":
        return cpu.batched_knn(ridsList, RList, QList, k, cores, qidsList=qidsList, neighbor_ids=neighbor_ids, neighbor_dist=neighbor_dist, gids=gids, repack=repack, n=n)
   if env == "GPU":
        #TODO: ADD FILKNN here

def dense_build(P):
    return cpu.dense_build(P)

def merge_neighbors(a, b, k, loc="HOST", cores=8):
    """Merge nearest neighbor results from different trees

    Arguments:
        a, b -- nearest neighbor results of the tuple form (neighbor_list, neighbor_dist)
        k    -- number of nearest neighbors to keep

    Note/Warning: 'a' should be changed during the course of this function

    Return:
        (a_list, a_dist) -- updated a
        changes -- number of updates made to the nearest neihgbor list a
    """

    if loc == "HOST":
        return cpu.merge_neighbors(a, b, k, cores)
    elif loc == "GPU":
        out = gpu.merge_neighbors(a, b, k, device)

    return out

def similarity_check(a, b):
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

    for i in range(Na):
        truth_sim[i] = np.mean(a_dist[i, :])
        approx_sim[i] = np.mean(b_dist[i, :])

    return truth_sim, approx_sim


def accuracy_stride(truth, approx, k_list=None, id_only=False,):

    accuracy_list = []
    error_list = []
    sim_list = []

    truth_list = truth[0]
    truth_dist = truth[1]

    approx_list = approx[0]
    approx_dist = approx[1]

    k_max = approx_dist.shape[1]

    if k_list is None:
        k_list = [2, 0.25*k_max, 0.5*k_max, 0.75*k_max, k_max]
        k_list = [int(i) for i in k_list]

    for k in k_list:
        k_truth = (truth_list[:, :k], truth_dist[:, :k])
        k_approx = (approx_list[:, :k], approx_dist[:, :k])
        accuracy = check_accuracy(k_truth, k_approx, id_only)
        accuracy_list.append(accuracy[0])
        error_list.append(accuracy[1])
        sim_list.append(accuracy[2])

    return accuracy_list, error_list, sim_list


def check_accuracy(a, b):
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
    approx_id = b_list 
    approx_dist = b_dist 

    truth_id = a_list
    truth_dist = a_dist 
    
    err = 0.0
    for i in range(N):

        miss_array_id = [0 if approx_id[i, j] in truth_id[i, :] else 0 for j in range(k)]
        miss_array_dist = [1 if approx_dist[i, j] <= truth_dist[i, -1] else 0 for j in range(k)]

        err += np.sum(np.logical_or(miss_array_id, miss_array_dist))


    mean_sim = np.mean(approx_dist.ravel())
    last_array = np.abs(approx_dist[:, -1] - truth_dist[:, -1])/truth_dist[:, -1]
    mean_rel_err = np.mean(last_array)
    hit_rate = err/(Na*ka)

    return hit_rate, mean_rel_err, mean_sim

def reindex(val, index, copy_back=False, use_numpy=False):
    return cpu.reindex(val, index, copy_back=False, use_numpy=False)

def argsort(val, index=None, dtype=np.int32):
    if index is None:
        index = np.empty(len(val), dtype=dtype)
    cpu.argsort(index, val)
    return index

def interval(starts, sizes, index, nleaves, leaf_ids):
    return cpu.interval(starts, sizes, index, nleaves, leaf_ids) 

def dist_select(rank, k, data, ids, comm):
    return cpu.dist_select(rank, k, data, ids, comm)

def cpu_sparse_knn(gids, X, levels, ntrees, k, blocksize, cores=8):
    return cpu.sparse_knn(gids, X, levels, ntrees, k, blocksize, cores)

def cpu_sparse_knn_3(gids, ptr, idx, val, nnz, levels, ntrees, k, blocksize, n, d, cores=8):
    return cpu.sparse_knn_3(gids, ptr, idx, val, nnz, levels, ntrees, k, blocksize, cores, n, d)

def gpu_sparse_knn(gids, X, levels, ntrees, k, blockleaf, blocksize, device):
    return gpu_sparse.sparse_knn(gids, X, levels, ntrees, k, blockleaf, blocksize, device)

def gpu_dense_knn(gids, X, levels, ntrees, k, blocksize, device):
    return gpu.dense_knn(gids, X, levels, ntrees, k, blocksize, device)


