# distutil: language = c++
#cython: boundscheck=False

import numpy as np
cimport numpy as np

import cupy as cp
from numba import cuda

import time

import cython

from primitives cimport *

def dense_knn(gids, X, levels, ntrees, k, blocksize, device):
   
    n, d = X.shape
    
    cdef int[:] hID = gids;
    cdef float[:, :] hP = X;

    cdef int[:, :] nID = np.zeros([n, k], dtype=np.int32) + -1
    cdef float[:, :] nDist = np.zeros([n, k], dtype=np.float32) + 1e38

    denknn( <int*> &hID[0], <float*> &hP[0,0], <int> n, <int> d, <int> levels, <int> ntrees, <int*> &nID[0, 0], <float*> &nDist[0,0], <int> k, <int> blocksize, <int> device);

    outID = np.asarray(nID)
    outDist = np.asarray(nDist)

    return (outID, outDist)

def sparse_knn(gids, X, levels, ntrees, k, blockleaf, blocksize, device):
    n, d = X.shape

    assert(n == len(gids))

    cdef int[:] hID = gids

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

cpdef merge_neighbors(a, b, k, device):

    I1 = a[0]
    D1 = a[1]

    I2 = b[0]
    D2 = b[1]

    cdef float[:, :] cD1 = D1;
    cdef float[:, :] cD2 = D2;

    cdef int[:, :] cI1 = I1;
    cdef int[:, :] cI2 = I2;

    cdef int cn = I1.shape[0]
    cdef int ck = k
    cdef int cdevice = device
    
    with nogil:
        merge_neighbors_python( <float*> &cD1[0, 0], <int*> &cI1[0, 0], <float*> &cD2[0, 0], <int*> &cI2[0, 0], cn, ck, ck, <int> cdevice)

    return (I1, D1)
