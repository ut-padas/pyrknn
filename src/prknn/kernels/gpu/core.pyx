import numpy as np
cimport numpy as np

import cupy as cp
from numba import cuda

import cython

from primitives cimport *


cpdef multileaf_knn(gidsList, RList, QList, k):
    
    cdef int nleaves = len(RList)
    cdef int cd = RList[0].shape[1];
    cdef int ck = k;

    cdef int blocksize = cd; 

    cRList = cp.zeros(nleaves, dtype=cp.uintp);
    cQList = cp.zeros(nleaves, dtype=cp.uintp);

    cqgidsList = cp.zeros(nleaves, dtype=cp.uintp);
    crgidsList = cp.zeros(nleaves, dtype=cp.uintp);

    NLList = []
    NDList = []

    cNL = cp.zeros(nleaves, dtype=cp.uintp);
    cND = cp.zeros(nleaves, dtype=cp.uintp);

    cdef int[:] cns = np.zeros(nleaves, dtype=np.int32)
    cdef int[:] cms = np.zeros(nleaves, dtype=np.int32)

    for i in range(nleaves):
        localR = RList[i]
        localQ = QList[i]

        localn = localR.shape[0];
        localm = localQ.shape[0];

        cRList[i] = <np.uintp_t> (<long> localR.data.mem.ptr)
        cQList[i] = <np.uintp_t> (<long> localQ.data.mem.ptr)

        rgids = cp.copy(gidsList[i])
        
        crgidsList[i] = <np.uintp_t>(<long> rgids.data.mem.ptr)
  
        cns[i] = localn;
        cms[i] = localm;

    total_queries = np.sum(cms);

    print(cRList)
    print("Rloc", cRList.data.ptr)
    print("Rloc[0]", cRList[0].data.ptr)
    print(cQList)
    print(crgidsList)

    NL = cp.zeros((total_queries, k), dtype=cp.int32)
    ND = cp.zeros((total_queries, k), dtype=cp.float32)

    prefix_n = np.cumsum(cms) #do prefix sum

    #This shouldn't be making copies
    for i in range(nleaves):
        start = prefix_n[i-1] if i > 0 else 0
        stop  = prefix_n[i]
        stride = stop - start
        
        localNL = NL[start:stop, ...]
        localND = ND[start:stop, ...]

        NLList.append(localNL)
        NDList.append(localND)

        cND[i] = <np.uintp_t>(<long>(localND.data.ptr))
        cNL[i] = <np.uintp_t>(<long>(localND.data.ptr))

    cdef long ptr_cNL = <long> cNL.data.ptr
    cdef long ptr_cND = <long> cND.data.ptr

    cdef long ptr_cRList = <long> cRList.data.ptr
    cdef long ptr_cQList = <long> cQList.data.ptr
    cdef long ptr_crgidsList = <long> crgidsList.data.ptr
    print("Starting GPU Kernel")
    with nogil:
        knn_gpu(<float**> ptr_cRList, <float**> ptr_cQList, <int**> ptr_crgidsList, <float**>ptr_cND, <int **> ptr_cNL, <int> nleaves, <int> cns[0], <int> cd, <int> ck, <int> blocksize)
    print("Finished GPU Kernel")
    return (NLList, NDList)
    
        
def merge_neighbors(a, b, k):

    D1 = a[0]
    I1 = a[1]
    D2 = b[0]
    I2 = b[1]

    cdef long cD1;
    cdef long cD2;
    cdef long cI1;
    cdef long cI2;

    cD1 = <long> D1.data.mem.ptr
    cD2 = <long> D2.data.mem.ptr

    cI1 = <long> I1.data.mem.ptr
    cI2 = <long> I2.data.mem.ptr

    cdef int ck = k;
    cdef int cm = D1.shape[0]
    cdef int cn = D1.shape[1]

    cdef float* ptr_cD1 = <float*> cD1
    cdef float* ptr_cD2 = <float*> cD2
    cdef int* ptr_cI1 = <int*> cI1
    cdef int* ptr_cI2 = <int*> cI2
    with nogil:
        merge_neighbors_gpu( ptr_cD1, ptr_cI1, ptr_cD2, ptr_cI2, cm, cn, ck); 

    return (D1, I1)

             
        





















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


#Ignoring any type checking (#TODO: Do this?)
def median(a):
    cdef size_t N = len(a)
    cdef float* c_a
    cdef float median
    with cp.cuda.Device(a.data.device.id):
        temp_a = <long> a.data.mem.ptr
        c_a = <float *> temp_a

        median = <float> device_kelley_cutting(c_a, N)

    return median



#For go type checking
def reduce_float(a):
    cdef size_t N = len(a)
    cdef float* c_out
    cdef float* c_a
    
    with cp.cuda.Device(a.data.device.id):
        out = cp.zeros(1, dtype=cp.float32)
        temp_out = <long> out.data.mem.ptr
        temp_a = <long> a.data.mem.ptr

        c_a = <float *> temp_a
        c_out = <float *> temp_out

        device_reduce_warp_atomic(c_a, c_out, N)

    return out


