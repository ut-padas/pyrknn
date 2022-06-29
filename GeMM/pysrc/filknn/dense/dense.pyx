


from libcpp.vector cimport vector
from libcpp.string cimport string


import cupy as cp
import numpy as np
import cython
import time 



cdef extern from "dfiknn.h" nogil:
  #cdef void dfi_leafknn(float *data, int *G_Id, int M, int leaves, int k, float *knn, int *knn_Id, int dim);
  void dfi_leafknn(float *data, int *G_Id, int M, int leaves, int k, float *knn, int *knn_Id, int dim, int deviceId);
  void memcheck();

cdef wrapper_dense(size_t dataPtr, size_t gidPtr, int n, int leaves, int k, size_t NhbdPtr, size_t IdPtr, int dim, int deviceId):
    cdef int dev = deviceId
    cdef int d = dim
    cdef int npoints = n
    cdef int nleaves = leaves
    cdef int nk = k
    with nogil:
        dfi_leafknn(<float*> dataPtr, <int*> gidPtr, npoints, nleaves, nk, <float*> NhbdPtr, <int*> IdPtr, <int> dim, <int> dev)


def memory():
    with nogil:
        memcheck()

@cython.boundscheck(False)
@cython.wraparound(False)
def py_dfiknn(gids, X, leaves, k, knnidx, knndis, dim, deviceId):

  n,_ = X.shape

  assert(n == len(gids))

  tic = time.time()
  X_l = X[gids, :]
  toc = time.time() - tic
  #print("Data permutation : %.4f sec "%toc)

  hID = gids.data.ptr
  data = X_l.data.ptr
  
  nID = knnidx.data.ptr
  nDist = knndis.data.ptr

  wrapper_dense(data, hID, n, leaves, k, nDist, nID, dim, deviceId) 

  del X_l

  return (knnidx, knndis)
