# distutils: language = c++


from libcpp.vector cimport vector

cdef extern from "impl/exact/exact.hpp" nogil:
    cdef void exact_knn(int, int, int, int*, int*, float*, int, int, int, int*, int*, float*, int*, int, int*, float*) except + 

#cdef extern from "impl/sparse/spknn.hpp" nogil:
#    cdef void spknn(int*, int*, int*, float*, int, int, int, int*, float*, int, int, int, int, int) except +
#    cdef void spknn(unsigned int*, int*, int*, float*, unsigned int, unsigned int, unsigned int, unsigned int*, float*, int, int, int, int, int) except +

cdef extern from "impl/primitives_shared.hpp" nogil:
    #cdef void GSKNN[T](int *rgids, int *qgids, T *R, T *Q, int n, int d, int m, int k, int *neighbor_list, T *neighbor_dist) except +
    #cdef void batchedGSKNN[T](int **rgids,int **qgids, T **R, T **Q, int *n, int d, int *m, int k, int **neighbor_list, T **neighbor_dist, int nleaves, int cores) except +
    cdef void direct_knn_base(float* R, float* Q, int n, int m, int d, int k, int* nids, float* ndists, int blocksize) except +
    cdef void build_tree(float* X, unsigned int* order, unsigned int* firstPt, const unsigned int n, const size_t L) except +
    cdef void merge_neighbor_cpu[T](T* D1,unsigned int* I1, T* D2, unsigned int* I2, unsigned int n, int k, int cores) except +
    cdef void find_interval(int* starts, int* sizes, unsigned char* index, int len, int nleaves, unsigned char* leaf_ids) except +
    cdef void arg_sort[T1, T2](T1* idx, T2* val, size_t length) except +
    cdef void reindex_1D[T1, T2](T1* idx, T2* val, size_t length, T2* buf) except+
    cdef void reindex_2D[T1, T2](T1* idx, T2* val, size_t length, size_t dim, T2* buf) except +
    cdef void map_1D[T1, T2](T1* idx, T2* val, size_t length, T2* buf) except+
    cdef void map_2D[T1, T2](T1* idx, T2* val, size_t length, size_t dim, T2* buf) except +
    cdef void bin_queries[T1, T2](size_t n, int levels, T2* proj, T2* medians, T1* idx, T2* buf) except + 
    cdef void bin_queries_pack[T1, T2](size_t n, int levels, T2* proj, T2* medians, T1* idx, T2* buf) except + 
    #cdef void bin_queries_simd(size_t n, int levels, float* proj, float* medians, int* idx, float* buf) except + 

