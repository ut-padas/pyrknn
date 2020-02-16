
cdef extern from "impl/primitives.hpp" nogil:
    cdef void vector_add(float *out, float *a, float* b, size_t n)
