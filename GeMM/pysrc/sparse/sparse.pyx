


from libcpp.vector cimport vector
from libcpp.string cimport string

from scipy import sparse
import cython
import cupy as cp
from cupyx.scipy.sparse import csr_matrix
from cupyx.scipy.sparse import eye
import time


cdef extern from "sfiknn.h" nogil:
  void sfi_leafknn(int *R, int *C, float *V, int *G_Id, int M, int leaves, int k, float *knn, int *knn_Id);

cdef wrapper_sparse(size_t RowPtr, size_t ColPtr, size_t ValPtr, size_t IDPtr, int n, int leaves, int k, size_t NDistPtr, size_t NIdPtr):
  sfi_leafknn(<int*>RowPtr, <int*>ColPtr, <float*> ValPtr, <int*> IDPtr, n, leaves, k, <float*> NDistPtr, <int*> NIdPtr)


@cython.boundscheck(False)
@cython.wraparound(False)
def py_sfiknn(gids, X, leaves, k, knndis, knnidx):

  n,_ = X.shape

  assert(n == len(gids))


  tic = time.time()
  X_c = X[gids, :]

  toc = time.time() - tic

  print("Data permutation %.4f"%toc)

  hID = gids.data.ptr
  vals = X.data.ptr
  idx = X.indices.data.ptr
  rowptr = X.indptr.data.ptr
  nID = knnidx.ravel().data.ptr
  nDist = knndis.ravel().data.ptr
  
  wrapper_sparse(rowptr, idx, vals, hID, n, leaves, k, nDist, nID)

  return (knnidx, knndis)
