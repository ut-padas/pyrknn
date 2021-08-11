from libcpp.vector cimport vector
from libcpp.string cimport string

cdef extern from "simple.hpp" nogil:
    cdef void gpu_knn(int *R, int *C, float *V, int *G_Id, int M, int leaves, int k, float *knn, int *knn_Id, int max_nnz);



