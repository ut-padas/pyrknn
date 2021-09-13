


from libcpp.vector cimport vector
from libcpp.string cimport string

import numpy as np
cimport numpy as np
from scipy import sparse
import cython
import cupy as cp

cdef extern from "FIKNN_dense.h" nogil:
  cdef void dfi_leafknn(float *data, int *G_Id, int M, int leaves, int k, float *knn, int *knn_Id, int dim);


#def py_FIKNN_sparse(R, C, V, GId, M, leaves, k, knndis_np, knnidx_np, maxnnz):
@cython.boundscheck(False)
@cython.wraparound(False)
def py_dfiknn(gids, X, leaves, k, knndis, knnidx, dim):

  n,_ = X.shape

  assert(n == len(gids))


  '''
  I = eye(X.shape[0]).tocsr()
  I=I[gids, :]
  I = csr_matrix(I, dtype = np.float32)
  X_csr = I*X

  #gids = np.arange(n, dtype = np.int32)
  
  '''
  #X_csr = X



  cdef int[:] hID = np.asarray(gids, dtype = np.int32)

  cdef float[:] data =  np.asarray(X, dtype = np.float32)

  cdef int c_n = n
  cdef int c_k = k
  cdef int c_leaves = leaves
  cdef int c_dim = dim

  cdef int[:] nID = np.asarray(knnidx, dtype = np.int32)
  cdef float[:] nDist = np.asarray(knndis, dtype = np.float32)

  with nogil:
        dfi_leafknn(&data[0], &hID[0], c_n, c_leaves, c_k, &nDist[0], &nID[0], c_dim) 

  outID = np.asarray(nID)
  outDist = np.asarray(nDist) 

  outID = np.reshape(outID, (n,k))
  outDist = np.reshape(outDist, (n,k))
  

  return (outID, outDist)
