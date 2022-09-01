


from libcpp.vector cimport vector
from libcpp.string cimport string

import cython
import cupy as cp
from cupyx.scipy.sparse import csr_matrix
from cupyx.scipy.sparse import eye
import time


cdef extern from "sfiknn.h" nogil:
  void sfi_leafknn(int *R, int *C, float *V, int *G_Id, int M, int leaves, int k, float *knn, int *knn_Id, int deviceId);

cdef wrapper_sparse(size_t RowPtr, size_t ColPtr, size_t ValPtr, size_t IDPtr, int n, int leaves, int k, size_t NDistPtr, size_t NIdPtr, deviceId):
    cdef int dev = deviceId;
    with nogil:
        sfi_leafknn(<int*>RowPtr, <int*>ColPtr, <float*> ValPtr, <int*> IDPtr, n, leaves, k, <float*> NDistPtr, <int*> NIdPtr, <int> dev)


@cython.boundscheck(False)
@cython.wraparound(False)
def py_sfiknn(gids, X, leaves, k, knndis, knnidx, deviceId):

  n,_ = X.shape
  assert(n == gids.shape[0])

  tic = time.time()
  X_c = X[gids, :]
  
  toc = time.time() - tic

  print("Data permutation %.4f"%toc)
  print("X", X_c)
  mempool = cp.get_default_memory_pool()
  mempool.free_all_blocks()
  hID = gids.data.ptr
  
  vals = X_c.data.data.ptr
  idx = X_c.indices.data.ptr
  rowptr = X_c.indptr.data.ptr
  nID = knnidx.data.ptr
  nDist = knndis.data.ptr
    
  wrapper_sparse(rowptr, idx, vals, hID, n, leaves, k, nDist, nID, deviceId)
 

  return (knnidx, knndis)
