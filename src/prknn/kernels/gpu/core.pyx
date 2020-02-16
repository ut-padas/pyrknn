import numpy as np
cimport numpy as np

import cupy as cp
cimport cupy as cp

import cython

from primitives cimport *

#Theres not really a point to this. It's just a compilation test and helpful example of how to format a function. 
def add_vectors(a, b):
    """Adds two vectors on the gpu. """

    cdef size_t N = len(a):
    cdef long device_a
    cdef long device_b
    
    #Assert that the vectors are the same length
    assert(N == len(b))

    if isinstance(array, (np.ndarray, np.generic)):
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
        with cupy.cuda.Device(device_a):
            out = cp.zeros(N)
            temp_out = <long> out.data.mem.ptr
            c_out = <float *> temp_out

            vector_add(c_a, &c_b, N)

    return out
