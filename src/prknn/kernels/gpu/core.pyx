import numpy as np
cimport numpy as np

#import numpy as cp
import cupy as cp
from numba import cuda

import time

import cython

from primitives cimport *

"""
Assume:
gids is a cp array (int32)
csr_data is a cpyx.csr datatype (float32, int32, int32)
current_offset is a cp array(int32)
new_offset is a cp array(int32)
dim is an int
cpdef sparse_build_level(gids, csr_data, current_offset, new_offset, dim):

    built_t = time.time()


    cdef int current_nleaves = len(current_offset)
    cdef int next_nleaves = len(new_offset)
    cdef int nnz = csr_data.count_nonzero()

    cdef int n = csr_data.shape[0]
    cdef int d = dim

    #Create pointer to gids array
    cdef long ptr_gids = <long> gids.data.ptr


    #Create pointers to CSR data (data, indicies, indptr)
    local_data = csr_data.data
    local_indices = csr_data.indicies
    local_indptr = csr_data.indptr

    cdef long ptr_data = <long> local_data.data.ptr
    cdef long ptr_indices = <long> local_indices.data.ptr
    cdef long ptr_indptr = <long> csr_data.indptr


    #Create pointers to Segment Heads (Current, and Next)

    cdef long ptr_seghead = <long> current_offset.data.ptr
    cdef long ptr_segHeadNext = <long> new_offset.data.ptr

    #Allocate and create pointers to random array (value X)
    X = cp.random.rand((current_nleaves*d), dtype=np.float32)
    cdef long ptr_X = <long> X.data.ptr

    #Allocate and create pointer to median array
    median = cp.zeros(current_nleaves, dtype=np.float32)
    cdef long ptr_median = <long> median.data.ptr

    #with nogil:
        #Call out to Chao's function

    return (csr_data, median)

"""
"""
Assume:
gidsList is a list of cp arrays (int32)
RList is a list of csr_matrices (float32, int32, int32)
Qlist is a list of csr_matrices (float32, int32, int32)
k int
"""
#cpdef sparse_batched_knn(gidsList, RList, QList, k):
#    t_batched =   



cpdef batched_knn(gidsList, RList, QList, k):

    book_t = time.time()    

    cdef int nleaves = len(RList)

    #print("nleaves", nleaves)

    cdef int cd = RList[0].shape[1];
    cdef int ck = k;

    cdef int blocksize = 128; 

    cdef size_t[:] cRList = np.zeros(nleaves, dtype=np.uintp);
    cdef size_t[:] cQList = np.zeros(nleaves, dtype=np.uintp);

    cdef size_t[:] cqgidsList = np.zeros(nleaves, dtype=np.uintp);
    cdef size_t[:] crgidsList = np.zeros(nleaves, dtype=np.uintp);

    NLList = []
    NDList = []
    gidsList_temp = []
    
    cdef size_t[:] cNL = np.zeros(nleaves, dtype=np.uintp);
    cdef size_t[:] cND = np.zeros(nleaves, dtype=np.uintp);

    cdef int[:] cns = np.zeros(nleaves, dtype=np.int32)
    cdef int[:] cms = np.zeros(nleaves, dtype=np.int32)

    for i in range(nleaves):
        localR = RList[i]
        localQ = QList[i]

        localn = localR.shape[0];
        localm = localQ.shape[0];

        cRList[i] = <np.uintp_t> (<long> localR.data.ptr)
        cQList[i] = <np.uintp_t> (<long> localQ.data.ptr)

        rgids = cp.copy(gidsList[i])
        gidsList_temp.append(rgids)

        crgidsList[i] = <np.uintp_t>(<long> rgids.data.ptr)
  
        cns[i] = localn;
        cms[i] = localm;
        #print(localn)

    total_queries = np.sum(cms);

    NL = cp.zeros((total_queries, k), dtype=cp.int32)
    ND = cp.zeros((total_queries, k), dtype=cp.float32)

    prefix_n = np.cumsum(cms) #do prefix sum
    #print(prefix_n)

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
        cNL[i] = <np.uintp_t>(<long>(localNL.data.ptr))

    book_t = time.time() - book_t
    print("Bookkeeping to C++ took ",book_t) 
    cdef int device = <int> NL.data.device.id
    print("Running batch on gpu:", device)
    with nogil:
        knn_gpu(<float**> &cRList[0], <float**> &cQList[0], <int**> &crgidsList[0], <float**> &cND[0], <int **> &cNL[0], <int> nleaves, <int> cns[0], <int> cd, <int> ck, <int> blocksize, <int> device)
    
    return (NLList, NDList)
    
        
def merge_neighbors(a, b, k):

    merge_t = time.time()

    I1 = a[0]
    D1 = a[1]

    I2 = b[0]
    D2 = b[1]

    cdef long cD1;
    cdef long cD2;

    cdef long cI1;
    cdef long cI2;

    cD1 = <long> D1.data.ptr
    cD2 = <long> D2.data.ptr

    cI1 = <long> I1.data.ptr
    cI2 = <long> I2.data.ptr

    cdef int ck = k;

    cdef int cm = D1.shape[0] #Should be N
    cdef int cn = D1.shape[1] #Should be k

    cdef float* ptr_cD1 = <float*> cD1
    cdef float* ptr_cD2 = <float*> cD2

    cdef int* ptr_cI1 = <int*> cI1
    cdef int* ptr_cI2 = <int*> cI2
    
    cdef int device = I1.data.device.id

    with nogil:
        merge_neighbors_gpu(ptr_cD1, ptr_cI1, ptr_cD2, ptr_cI2, cm, cn, ck, <int> device); 

    merge_t = time.time() - merge_t
    print("Merge time:", merge_t)

    return (I1, D1)

             
        





















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


