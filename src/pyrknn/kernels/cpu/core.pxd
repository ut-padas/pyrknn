# distutils: language = c++

from libcpp.vector cimport vector
cdef extern from "impl/exact/exact.hpp" nogil:
    cdef void exact_knn(int, int, int, int*, int*, float*, int, int, int, int*, int*, float*, int*, int, int*, float*) except + 

cdef extern from "impl/sparse/spknn.hpp" nogil:
    cdef void spknn(int*, int*, int*, float*, int, int, int, int*, float*, int, int, int, int, int) except +
    cdef void spknn(unsigned int*, int*, int*, float*, unsigned int, unsigned int, unsigned int, unsigned int*, float*, int, int, int, int, int) except +

cdef extern from "impl/primitives_shared.hpp" nogil:
    cdef void GSKNN[T](int *rgids, int *qgids, T *R, T *Q, int n, int d, int m, int k, int *neighbor_list, T *neighbor_dist) except +
    cdef void batchedGSKNN[T](int **rgids,int **qgids, T **R, T **Q, int *n, int d, int *m, int k, int **neighbor_list, T **neighbor_dist, int nleaves, int cores) except +
    cdef void build_tree(float* X, unsigned int* order, unsigned int* firstPt, const unsigned int n, const size_t L) except +
    cdef void merge_neighbor_cpu[T](T* D1,unsigned int* I1, T* D2, unsigned int* I2, unsigned int n, int k, int cores) except +
