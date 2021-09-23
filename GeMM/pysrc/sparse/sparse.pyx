


from libcpp.vector cimport vector
from libcpp.string cimport string

from scipy import sparse
import cython
import cupy as cp
from cupyx.scipy.sparse import csr_matrix
from cupyx.scipy.sparse import eye
import time


cdef extern from "sfiknn.h" nogil:
  cdef void sfi_leafknn(int *R, int *C, float *V, int *G_Id, int M, int leaves, int k, float *knn, int *knn_Id);


@cython.boundscheck(False)
@cython.wraparound(False)
def py_sfiknn(gids, X, leaves, k, knndis, knnidx, iternum):

  n,_ = X.shape

  assert(n == len(gids))


  tic = time.time()
  I = eye(X.shape[0], dtype=cp.float32).tocsr()
  I=I[gids, :]
  X = I*X

  toc = time.time() - tic

  print("Data permutation %.4f"%toc)

  cdef int[:] hID = cp.asarray(gids, dtype = cp.int32)

  cdef float[:] data =  cp.asarray(X.data, dtype = cp.float32)
  cdef int[:] idx =  cp.asarray(X.indices, dtype = cp.int32)
  cdef int[:] ptr =  cp.asarray(X.indptr, dtype = cp.int32)

  cdef int c_n = n
  cdef int c_k = k
  cdef int c_leaves = leaves

  cdef int[:] nID = cp.asarray(knnidx.ravel(), dtype = cp.int32)
  cdef float[:] nDist = cp.asarray(knndis.ravel(), dtype = cp.float32)
  cdef int c_iternum = iternum

  with nogil:
        sfi_leafknn(&ptr[0], &idx[0], &data[0], &hID[0], c_n, c_leaves, c_k, &nDist[0], &nID[0]) 

  knnidx = cp.reshape(cp.asarray(nID), (n,k))
  knndis = cp.reshape(cp.asarray(nDist), (n,k))


  return (knnidx, knndis)
