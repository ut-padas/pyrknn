# distutils: language = c++

from libcpp.vector cimport vector

cdef extern from "impl/primitives_shared.hpp" namespace "hmlp::combinatorics" nogil:
    cdef vector[T] sampleWithoutReplacement[T](int l, vector[T] v) except + 
    cdef vector[T] Sum[T,Allocator]( size_t d, size_t n, vector[T, Allocator] & X, vector[size_t] & gids ) except +
    cdef T Accumulate[T](vector[T] &, T & sum_glb) except +
    cdef vector[TB] Scan[TA,TB]( vector[TA] & ) except +
    cdef T Select[T]( size_t k, vector[T] &x ) except +
    cdef T Reduce[T](vector[T] &, T & sum_glb) except +
    # cdef vector[vector[uint64_t]] MedianThreeWaySplit[T]( vector[T] &v, T tol ) except +
    # cdef vector[vector[uint64_t]] MedianSplit[T](vector[T] &v) except +

