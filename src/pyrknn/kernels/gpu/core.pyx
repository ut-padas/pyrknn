# distutil: language = c++
#cython: boundscheck=False

import numpy as np
cimport numpy as np

import cupy as cp
from numba import cuda

import time

import cython

from core cimport *

def dense_knn(gids, X, levels, ntrees, k, blocksize, device):
   
    n, d = X.shape
    
    cdef int[:] hID = gids;
    cdef float[:, :] hP = X;

    cdef int[:, :] nID = np.zeros([n, k], dtype=np.int32) + -1
    cdef float[:, :] nDist = np.zeros([n, k], dtype=np.float32) + 1e38

    cdef int c_levels = levels 
    cdef int c_ntrees = ntrees 
    cdef int c_blocksize = blocksize 
    cdef int c_device = device 
    cdef int c_n = n 
    cdef int c_d = d
    cdef int c_k = k 

    with nogil:
        denknn( <int*> &hID[0], <float*> &hP[0,0], <int> c_n, <int> c_d, <int> c_levels, <int> c_ntrees, <int*> &nID[0, 0], <float*> &nDist[0,0], <int> c_k, <int> c_blocksize, <int> c_device);

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
