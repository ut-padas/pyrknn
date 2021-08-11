from core cimport *
import numpy as np
cimport numpy as np
import cupy as cp
import cython

@cython.boundscheck(False)
@cython.wraparound(False)
def py_gpuknn(R, C, V, G_Id, M, leaves, k, knn, knn_Id, max_nnz):

    cdef int[:] c_R
    cdef int[:] c_C
    cdef float[:] c_V
    cdef int[:] c_G_Id
    cdef float[:] c_knn
    cdef int[:] c_knn_Id
    
    cdef int c_max_nnz = max_nnz
    cdef int c_M = M
    cdef int c_leaves = leaves  
    cdef int c_k = k
    

    #cast to float
    c_R = np.asarray(R, dtype='int32')
    c_C = np.asarray(C, dtype='int32')
    c_V = np.asarray(V, dtype='float32')
    c_G_Id = np.asarray(G_Id, dtype='int32')
    c_knn = np.asarray(knn, dtype='float32')
    c_knn_Id = np.asarray(knn_Id, dtype='int32')
    '''
    c_R = cp.asarray(R, dtype='int32')
    c_C = cp.asarray(C, dtype='int32')
    c_V = cp.asarray(V, dtype='float32')
    c_G_Id = cp.asarray(G_Id, dtype='int32')
    c_knn = cp.asarray(knn, dtype='float32')
    c_knn_Id = cp.asarray(knn_Id, dtype='int32')
    #call out of c++ & cuda code
    ''' 
  
    gpu_knn(&c_R[0], &c_C[0], &c_V[0], &c_G_Id[0], c_M, c_leaves, c_k, &c_knn[0], &c_knn_Id[0], c_max_nnz)
    
    return np.asarray(c_knn), np.asarray(c_knn_Id)   
