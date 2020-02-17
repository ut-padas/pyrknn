import numpy as np
cimport numpy as np

import cupy as cp
from numba import cuda

import cython

from primitives cimport *

#Theres not really a point to this. It's just a compilation test and helpful example of how to format a function. 
def add_vectors(a, b):
    """Adds two vectors on the gpu. """

    cdef size_t N = len(a)
    cdef long device_a
    cdef long device_b
    
    #Assert that the vectors are the same length
    assert(N == len(b))

    if isinstance(a, (np.ndarray, np.generic)) or isinstance(b, (np.ndarray, np.generic)):
        raise Exception(" GPU `add_vectors` kernel requires data is already on the gpu. Please pass a cupy array.")
    else: #At the time of writing I'm actually not sure what the cupy type is...
        device_a = <long> a.data.device.id
        device_b = <long> b.data.device.id

        #Assert that the vectors are located on the same device
        assert(device_a == device_b)

        temp_a = <long> a.data.mem.ptr
        temp_b = <long> b.data.mem.ptr
        c_a = <float *> temp_a
        c_b = <float *> temp_b

        #allocate space for return array out
        with cp.cuda.Device(device_a):
            out = cp.zeros(N, dtype=cp.float32)
            temp_out = <long> out.data.mem.ptr
            c_out = <float *> temp_out

            vector_add(c_out, c_a, c_b, N)

    return out

#Function to test is thrust is working for me
def sort(a):
    """Sort a vectors on the gpu. """

    cdef size_t N = len(a)
    cdef long device_a
    
    if isinstance(a, (np.ndarray, np.generic)):
        raise Exception(" GPU `sort` kernel requires data is already on the gpu. Please pass a cupy array.")
    else: #At the time of writing I'm actually not sure what the cupy type is...
        device_a = <long> a.data.device.id

        temp_a = <long> a.data.mem.ptr
        c_a = <float *> temp_a

        #TODO(p2): How to make sure that the device being called by thrust is the same as the one the device is on? (Can this be done on the python layer)
        thrust_sort(c_a, N, device_a)

    return a

@cuda.jit
def numba_add(x, y, out):
    start = cuda.grid(1)
    stride = cuda.gridsize(1)
    for i in range(start, x.shape[0], stride):
        out[i] = x[i] + y[i]

@cuda.jit
def numba_increment(a):
    """ Perform vector increment by 1 """

    pos = cuda.grid(1)
    if pos < a.size:
        a[pos] += 1
