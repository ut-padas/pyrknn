from libcpp cimport bool

cdef extern from "impl/dense/denknn.hpp" nogil:
    cdef void denknn(int*, float*, int, int, int, int, int*, float*, int, int, int) except +
    cdef void merge_neighbors_python(float*, int*, float*, int*, int, int, int, int) except +

cdef extern from "impl/sparse/spknn.hpp" nogil:
    cdef void spknn(int*, int*, int*, float*, int, int, int, int, int, int*, float*, int, int, int, int) except +
 
    

