from libcpp cimport bool

cdef extern from "impl/primitives.hpp" nogil:
    cdef void vector_add(float *out, float *a, float* b, size_t n)
    cdef void device_reduce_warp_atomic(float *, float *, size_t)
    cdef float device_kelley_cutting(float *arr, const size_t n)

cdef extern from "impl/chao/merge/merge_gpu.hpp" nogil:
    cdef void merge_neighbors_gpu(float *nborD1, int *nborI1, const float *nborD2, const int *nborI2, int m, int n, int k)

cdef extern from "impl/chao/knn_gpu.hpp" nogil:
    cdef void knn_gpu(float **, float**, int**, float **, int **, int, int, int, int, int)

#cdef extern from "impl/chao/sparse/knn_sparse_gpu.hpp" nogil:
#    cdef void find_knn(int *, int*, int*, float*, int, int, int, int*, int, int, int, int*, float*, int, int)
#    cdef void create_tree_next_level(int*, int*, int*, float*, int, int, int, int*, int*, int, float*, float*);
 
    

