# distutil: language = c++
#cython: boundscheck=False

import numpy as np
cimport numpy as np

import cupy as cp
from numba import cuda

import time

import cython

cdef extern from "impl/sparse/spknn.hpp" nogil:
    cdef void spknn(int*, int*, int*, float*, int, int, int, int, int, int*, float*, int, int, int, int) except +

cdef extern from "impl/sfiknn/sfiknn.hpp" nogil:
    cdef void FIKNN_sparse_gpu(int*, int*, float*, int*, int, int, int, float*, int*, int) except +


def sparse_knn(gids, X, levels, ntrees, k, blockleaf, blocksize, device):
    n, d = X.shape

    assert(n == len(gids))

    cdef int[:] hID = np.asarray(gids, dtype=np.int32)

    cdef float[:] data = X.data
    cdef int[:] idx = X.indices
    cdef int[:] ptr = X.indptr
    cdef int nnz = X.nnz
    cdef int[:, :] nID = np.zeros([n, k], dtype=np.int32)+ -1
    cdef float[:, :] nDist = np.zeros([n, k], dtype=np.float32) + 1e38

    cdef int c_n = n 
    cdef int c_d = d 
    cdef int c_nnz = nnz
    cdef int c_levels = levels 
    cdef int c_k = k 
    cdef int c_blockleaf = blockleaf 
    cdef int c_blocksize = blocksize
    cdef int c_device = device 
    cdef int c_ntrees = ntrees

    with nogil:
        spknn(<int*> &hID[0], <int*> &ptr[0], <int*> &idx[0], <float*> &data[0], <int> c_n, <int> c_d, <int> nnz, <int> c_levels, <int> c_ntrees, <int*> &nID[0, 0], <float*> &nDist[0,0], <int> c_k, <int> c_blockleaf, <int> c_blocksize, <int> c_device)

    outID = np.asarray(nID)
    outDist = np.asarray(nDist)

    return (outID, outDist)



def sparse_fiknn(gids, X, leaves, k):
     

    n,_ = X.shape

    assert(n == len(gids))
    
    maxnnz = cp.max(cp.diff(X.indptr))
    cdef int c_maxnnz = maxnnz

    cdef int[:] hID = np.asarray(gids, dtype=np.int32)

    cdef float[:] data = X.data
    cdef int[:] idx = X.indices
    cdef int[:] ptr = X.indptr
    cdef int[:] c_gids = gids

    cdef int [:] nID = np.zeros([n*k], dtype=np.int32) + -1
    cdef float [:] nDist = np.zeros([n*k], dtype = np.float32) + 1e38
 
    cdef int c_n = n
    cdef int c_k = k
    cdef int c_leaves = leaves

    with nogil:
        FIKNN_sparse_gpu(<int*> &ptr[0], <int*> &idx[0], <float*> &data[0], <int*> &c_gids[0], c_n, c_leaves, c_k, <float*> &nDist[0], <int*> &nID[0], c_maxnnz)
    
    outID = np.asarray(nID)
    outDist = np.asarray(nDist)
    
    return (outID, outDist)
    
