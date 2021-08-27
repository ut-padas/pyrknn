


from libcpp.vector cimport vector
from libcpp.string cimport string

import numpy as np
cimport numpy as np
from scipy import sparse
import cython
import cupy as cp
from scipy.sparse import csr_matrix
from scipy.sparse import eye

cdef extern from "FIKNN_sparse.h" nogil:
  cdef void FIKNN_sparse_gpu(int *R, int *C, float *V, int *G_Id, int M, int leaves, int k, float *knn, int *knn_Id, int max_nnz, int iternum);


#def py_FIKNN_sparse(R, C, V, GId, M, leaves, k, knndis_np, knnidx_np, maxnnz):
@cython.boundscheck(False)
@cython.wraparound(False)
def py_sfiknn(gids, X, leaves, k, knndis, knnidx, iternum):

  n,_ = X.shape

  assert(n == len(gids))


  '''
  I = eye(X.shape[0]).tocsr()
  I=I[gids, :]
  I = csr_matrix(I, dtype = np.float32)
  X_csr = I*X

  #gids = np.arange(n, dtype = np.int32)
  
  '''
  X_csr = X





  #maxnnz = cp.max(cp.diff(X.indptr))
  maxnnz = max(np.diff(X_csr.indptr))
  cdef int c_maxnnz = maxnnz

  cdef int[:] hID = np.asarray(gids, dtype = np.int32)

  cdef float[:] data =  np.asarray(X_csr.data, dtype = np.float32)
  cdef int[:] idx =  np.asarray(X_csr.indices, dtype = np.int32)
  cdef int[:] ptr =  np.asarray(X_csr.indptr, dtype = np.int32)

  cdef int c_n = n
  cdef int c_k = k
  cdef int c_leaves = leaves

  cdef int[:] nID = np.asarray(knnidx, dtype = np.int32)
  cdef float[:] nDist = np.asarray(knndis, dtype = np.float32)
  cdef int c_iternum = iternum

  with nogil:
        FIKNN_sparse_gpu(&ptr[0], &idx[0], &data[0], &hID[0], c_n, c_leaves, c_k, &nDist[0], &nID[0], c_maxnnz, c_iternum) 

  outID = np.asarray(nID)
  outDist = np.asarray(nDist) 

  outID = np.reshape(outID, (n,k))
  outDist = np.reshape(outDist, (n,k))
  

  return (outID, outDist)




