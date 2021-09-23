


from libcpp.vector cimport vector
from libcpp.string cimport string


import cupy as cp
import cython
import time 



cdef extern from "dfiknn.h" nogil:
  cdef void dfi_leafknn(float *data, int *G_Id, int M, int leaves, int k, float *knn, int *knn_Id, int dim);


@cython.boundscheck(False)
@cython.wraparound(False)
def py_dfiknn(gids, X, leaves, k, knndis, knnidx, dim):

  n,_ = X.shape

  assert(n == len(gids))

  tic = time.time()
  X = X[gids, :]
  toc = time.time() - tic
  print("Data permutation : %.4f "%toc)
  print(gids.__cuda_array_interface__)
  a = cp.cuda.texture.ChannelFormatDescriptor(gids)

  cdef int[:] hID = gids

  cdef float[:] data =  X.ravel()

  cdef int c_n = n
  cdef int c_k = k
  cdef int c_leaves = leaves
  cdef int c_dim = dim

  cdef int[:] nID = knnidx.ravel()
  cdef float[:] nDist = knndis.ravel()

  with nogil:
        dfi_leafknn(&data[0], &hID[0], c_n, c_leaves, c_k, &nDist[0], &nID[0], c_dim) 

  knnidx = cp.reshape(cp.asnumpy(nID), (n,k))
  knndis = cp.reshape(cp.asnumpy(nDist), (n,k))


  return (knnidx, knndis)




