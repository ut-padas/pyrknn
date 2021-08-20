

from libcpp.vector cimport vector
from libcpp.string cimport string


cdef extern from "sfiknn.hpp" nogil:
  cdef void FIKNN_sparse_gpu(int *R, int *C, float *V, int *G_Id, int M, int leaves, int k, float *knn, int *knn_Id, int max_nnz);



import numpy as np
cimport numpy as np
import cython


@cython.boundscheck(False)
@cython.wraparound(False)

def sfiknn(R, C, V, GId, M, leaves, k, knndis_np, knnidx_np, maxnnz):

  cdef int c_M = M
  cdef int c_leaves = leaves
  cdef int c_k = k
  cdef int c_maxnnz = maxnnz

  
  cdef int[:] c_R 
  cdef int[:] c_C 
  cdef float[:] c_V
  cdef float[:] c_knndis_np
  cdef int[:] c_knnidx_np
  cdef int[:] c_GId  
   
  c_R = np.asarray(R, dtype='int32')
  c_C = np.asarray(C, dtype='int32')
  c_V = np.asarray(V, dtype='float32')
  c_GId = np.asarray(GId, dtype='int32')

  c_knndis_np = np.asarray(knndis_np, dtype='float32')
  c_knnidx_np = np.asarray(knnidx_np, dtype='int32')
  
  
  FIKNN_sparse_gpu(&c_R[0], &c_C[0], &c_V[0], &c_GId[0], c_M, c_leaves, c_k, &c_knndis_np[0], &c_knnidx_np[0], c_maxnnz)


  return np.asarray(c_knndis_np), np.asarray(c_knnidx_np)

