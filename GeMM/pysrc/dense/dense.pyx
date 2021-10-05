


from libcpp.vector cimport vector
from libcpp.string cimport string


import cupy as cp
import numpy as np
import cython
import time 



cdef extern from "dfiknn.h" nogil:
  #cdef void dfi_leafknn(float *data, int *G_Id, int M, int leaves, int k, float *knn, int *knn_Id, int dim);
  void dfi_leafknn(float *data, int *G_Id, int M, int leaves, int k, float *knn, int *knn_Id, int dim, int deviceId);


cdef wrapper_dense(size_t dataPtr, size_t gidPtr, int n, int leaves, int k, size_t NhbdPtr, size_t IdPtr, int dim, int deviceId):
  dfi_leafknn(<float*> dataPtr, <int*> gidPtr, n, leaves, k, <float*> NhbdPtr, <int*> IdPtr, dim, deviceId)



@cython.boundscheck(False)
@cython.wraparound(False)
def py_dfiknn(gids, X, leaves, k, knnidx, knndis, dim, deviceId):

  n,_ = X.shape

  assert(n == len(gids))

  tic = time.time()
  X = X[gids, :]
  toc = time.time() - tic
  #print("Data permutation : %.4f sec "%toc)

  hID = gids.data.ptr
  data = X.ravel().data.ptr
  
  nID = knnidx.ravel().data.ptr
  nDist = knndis.ravel().data.ptr

  wrapper_dense(data, hID, n, leaves, k, nDist, nID, dim, deviceId) 

  return (knnidx, knndis)
