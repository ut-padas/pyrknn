# distutils: language = c++

from libcpp.vector cimport vector

cdef extern from "impl/exact/exact.hpp" nogil:
    cdef void exact_knn(int, int, int, int*, int*, float*, int, int, int, int*, int*, float*, int*, int, int*, float*) except + 

cdef extern from "impl/sparse/spknn.hpp" nogil:
    cdef void spknn(int*, int*, int*, float*, int, int, int, int*, float*, int, int, int, int, int) except +
    cdef void spknn(unsigned int*, int*, int*, float*, unsigned int, unsigned int, unsigned int, unsigned int*, float*, int, int, int, int, int) except +

cdef extern from "impl/primitives_shared.hpp" nogil:
    cdef vector[T] sampleWithoutReplacement[T](int l, vector[T] v) except + 
    cdef vector[T] Sum[T,Allocator]( size_t d, size_t n, vector[T, Allocator] & X, vector[size_t] & gids ) except +
    cdef T Accumulate[T](vector[T] &, T & sum_glb) except +
    cdef vector[TB] Scan[TA,TB]( vector[TA] & ) except +
    cdef T Select[T]( size_t k, vector[T] &x ) except +
    cdef T Reduce[T](vector[T] &, T & sum_glb) except +
    # cdef vector[vector[uint64_t]] MedianThreeWaySplit[T]( vector[T] &v, T tol ) except +
    # cdef vector[vector[uint64_t]] MedianSplit[T](vector[T] &v) except +
    cdef void directKLowMem[T](int* gids, T *R, T*Q, int n, int d, int m, int k, int *neighbor_list, T *neighbor_dist) except +
    cdef void GSKNN[T](int *rgids, int *qgids, T *R, T *Q, int n, int d, int m, int k, int *neighbor_list, T *neighbor_dist) except + 
    cdef void blockedGSKNN[T](int *rgids, int *qgids, T *R, T *Q, int n, int d, int m, int k, int *neighbor_list, T *neighbor_dist) except + 
    cdef void batchedDirectKNN[T](int **gids, T **R, T **Q, int *n, int d, int *m, int k, int **neighbor_list, T **neighbor_dist, int nleaves) except +
    cdef void batchedGSKNN[T](int **rgids,int **qgids, T **R, T **Q, int *n, int d, int *m, int k, int **neighbor_list, T **neighbor_dist, int nleaves, int cores) except +
    cdef void batchedRef[T](int **rgids,int **qgids, T **R, T **Q, int *n, int d, int *m, int k, int **neighbor_list, T **neighbor_dist, int nleaves) except +
    cdef void choose2(int *out, int N) except +
    cdef void quickselect[T](T *array, int N, int k) except +
    cdef void merge_neighbor_cpu[T](T* D1,unsigned int* I1, T* D2, unsigned int* I2, unsigned int n, int k, int cores) except +
