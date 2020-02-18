
cdef extern from "impl/primitives.hpp" nogil:
    cdef void vector_add(float *out, float *a, float* b, size_t n)
    cdef void device_reduce_warp_atomic(float *, float *, size_t)
    cdef float device_kelley_cutting(float *arr, const size_t n)
