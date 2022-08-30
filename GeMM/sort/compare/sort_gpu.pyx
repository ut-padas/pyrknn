


from libcpp.vector cimport vector
from libcpp.string cimport string

import numpy as np
cimport numpy as np
import cython
import cupy as cp

cdef extern from "SortMerge.h" nogil:
  cdef void SortInterface(float *data, float *Nhbd, int n_q, int l, int k);


#def py_FIKNN_sparse(R, C, V, GId, M, leaves, k, knndis_np, knnidx_np, maxnnz):
@cython.boundscheck(False)
@cython.wraparound(False)
def sort_gpu(data, n_q, l, k):

  cdef float[:] c_data =  np.asarray(np.ravel(data, 'C'), dtype = np.float32)

  cdef int c_n_q = n_q
  cdef int c_k = k
  cdef int c_l = l

  cdef float[:] Nhbd = np.zeros((n_q*k), dtype=np.float32) + 1e30 

  with nogil:
    SortInterface(&c_data[0], &Nhbd[0], c_n_q, c_l, c_k) 

  outNhbd = np.asarray(Nhbd)

  outNhbd = np.reshape(outNhbd, (n_q,k))
  

  return outNhbd
