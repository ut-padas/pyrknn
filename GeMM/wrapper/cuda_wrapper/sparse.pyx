


from libcpp.vector cimport vector
from libcpp.string cimport string

import numpy as np
cimport numpy as np

import cython
import cupy as cp

cdef extern from "FIKNN_sparse.h" nogil:
  cdef void FIKNN_sparse_gpu(int *R, int *C, float *V, int *G_Id, int M, int leaves, int k, float *knn, int *knn_Id, int max_nnz);


#def py_FIKNN_sparse(R, C, V, GId, M, leaves, k, knndis_np, knnidx_np, maxnnz):
def py_sfiknn(gids, X, leaves, k, knndis, knnidx):

  n,_ = X.shape

  assert(n == len(gids))

  #maxnnz = cp.max(cp.diff(X.indptr))
  maxnnz = max(np.diff(X.indptr))
  cdef int c_maxnnz = maxnnz

  cdef int[:] hID = gids

  cdef float[:] data =  X.data
  cdef int[:] idx =  X.indices
  cdef int[:] ptr =  X.indptr

  cdef int c_n = n
  cdef int c_k = k
  cdef int c_leaves = leaves

  cdef int[:] nID = knnidx
  cdef float[:] nDist = knndis

  with nogil:
        FIKNN_sparse_gpu(&ptr[0], &idx[0], &data[0], &hID[0], c_n, c_leaves, c_k, &nDist[0], &nID[0], c_maxnnz) 

  outID = np.asarray(nID)
  outDist = np.asarray(nDist) 

  outID = np.reshape(outID, (n,k))
  outDist = np.reshape(outDist, (n,k))
  

  return (outID, outDist)




