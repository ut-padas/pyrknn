# distutil: language = c++
#cython: boundscheck=False

import numpy as np
cimport numpy as np

import cupy as cp
from numba import cuda

import time

import cython

from primitives_sparse cimport *

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

    spknn(<int*> &hID[0], <int*> &ptr[0], <int*> &idx[0], <float*> &data[0], <int> n, <int> d, <int> nnz, <int> levels, <int> ntrees, <int*> &nID[0, 0], <float*> &nDist[0,0], <int> k, <int> blockleaf, <int> blocksize, <int> device)

    outID = np.asarray(nID)
    outDist = np.asarray(nDist)

    return (outID, outDist)
