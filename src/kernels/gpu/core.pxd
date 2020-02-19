from libcpp.vector cimport vector
from libcpp.string cimport string

cdef extern from "simple.hpp" nogil:
    cdef void addition(float *out, float *a, float *b, const int N)
    cdef void Exclusive_scan(int *array_out, int *array_in, int N)
    cdef void Sort(float *array_in, int N)
    cdef void Sort_by_key(float *keys, float *values, int N)
    #cdef void Partition(int *array_in, int N, int i)
