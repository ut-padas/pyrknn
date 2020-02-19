from core cimport *
import numpy as np
cimport numpy as np
#from numba import cuda
import cython
import cupy as cp

@cython.boundscheck(False)
@cython.wraparound(False)
def add_vectors(a, b):
    cdef int N = len(a)
    
    assert(N == len(b))
    assert(isinstance(a, (np.ndarray, np.generic)))
    assert(isinstance(b, (np.ndarray, np.generic)))
    cdef float[:] c_a
    cdef float[:] c_b
    cdef float[:] c_out = np.zeros(N, dtype='float32')

    #cast to float
    c_a = np.asarray(a, dtype='float32')
    c_b = np.asarray(b, dtype='float32')

    #call out of c++ & cuda code
    addition(&c_out[0], &c_a[0], &c_b[0], N)
    return np.asarray(c_out)

        
@cython.boundscheck(False)
@cython.wraparound(False)
def exclusive_scan(l):
    cdef int N = len(l)
    
    #assert(isinstance(l, (np.ndarray, np.generic)))

    cdef long temp
    temp = <long> l.data.mem.ptr
    cdef int* in_ptr = <int *> temp
    # cdef int[:] c_in = cp.asarray(l, dtype='int32')
    c_out = cp.zeros(N,dtype='int32')
    temp = <long> c_out.data.mem.ptr
    cdef int* out_ptr = <int *> temp

    #c_out = cuda.device_array(N,dtype='int32')
    Exclusive_scan(out_ptr, in_ptr, N)
    return c_out


@cython.boundscheck(False)
@cython.wraparound(False)
def sort(l):
    '''
    Accept an array on GPU and pass it to Sort.
    '''
    cdef int N = len(l)
    cdef long temp
    temp = <long> l.data.mem.ptr
    cdef float* arr_ptr = <float*> temp

    Sort(arr_ptr, N)
    return
    

@cython.boundscheck(False)
@cython.wraparound(False)
def sort_by_key(keys, values):
    cdef int N = len(keys)
    cdef long temp
    temp = <long> keys.data.mem.ptr
    cdef float* keys_ptr = <float*> temp

    temp = values.data.mem.ptr
    cdef float* values_ptr = <float*> temp
    Sort_by_key(keys_ptr, values_ptr, N)
    return