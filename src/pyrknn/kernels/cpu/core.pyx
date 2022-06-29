# distutils: language = c++
#cython: boundscheck=False

import numpy as np
cimport numpy as np

#from core cimport *
from libcpp.vector cimport vector
from cython.operator cimport dereference as deref
from cython.parallel import prange, parallel
from cython.view cimport array as cvarray
import cython
cimport cython 

cimport mpi4py.MPI as MPI
cimport mpi4py.libmpi as libmpi
from mpi4py import MPI
#import time as MPI
import math
import time

cdef fused real:
    cython.float
    #cython.double
    #cython.int
    #cython.long

cdef fused index:
    cython.int
    #cython.long

#cdef fused real:
#    cython.char
#    cython.uchar
#    cython.short
#    cython.ushort
#    cython.int
#    cython.uint
#    cython.long
#    cython.ulong
#    cython.longlong
#    cython.ulonglong
#    cython.float
#    cython.double

#cdef fused index:
#    cython.char
#    cython.uchar
#    cython.short
#    cython.ushort
#    cython.int
#    cython.uint
#    cython.long
#    cython.ulong
#    cython.longlong
#    cython.ulonglong


def argsort(index[:] idx, real[:] val):
    cdef size_t c_length = len(idx)
    arg_sort(&idx[0], &val[0], c_length)


def reindex_1(real[:] val, index[:] idx, real[:] buf):
    cdef size_t c_length = len(idx)
    reindex_1D(&idx[0], &val[0], c_length, &buf[0])

def reindex_2(real[:, :] val, index[:] idx, real[:, :] buf):
    cdef size_t c_length = len(idx)
    cdef size_t c_dim = val.shape[1]
    reindex_2D(&idx[0], &val[0, 0], c_length, c_dim, &buf[0, 0])

def map_1(real[:] val, index[:] idx, real[:] buf):
    cdef size_t c_length = len(idx)
    map_1D(&idx[0], &val[0], c_length, &buf[0])

def map_2(real[:, :] val, index[:] idx, real[:, :] buf):
    cdef size_t c_length = len(idx)
    cdef size_t c_dim = val.shape[1]
    map_2D(&idx[0], &val[0, 0], c_length, c_dim, &buf[0, 0])

def reindex(val, index, copy_back=False, use_numpy=False):
    source_shape = val.shape
    target_length = len(index)
    target_dim = val.shape[1]
    
    print(target_length, target_dim, val.ndim, flush=True)

    if val.ndim < 2:
        buf = np.empty(target_length, dtype=val.dtype)
    elif val.ndim == 2:
        buf = np.empty((target_length, target_dim), dtype=val.dtype)
    else:
        raise Exception("Reindex function only implemented for ndim <= 2")

    if use_numpy:
        if copy_back:
            assert(target_length == source_shape[0])
            val[:] = val[index]
            return val
        else:
            return val[index]

    if copy_back == True:
        assert(target_length == source_shape[0])
        if val.ndim < 2:
            reindex_1(val, index, buf)
        elif val.ndim == 2:
            reindex_2(val, index, buf)
        return val
    else:
        if val.ndim < 2:
            map_1(val, index, buf)
        elif val.ndim == 2:
            map_2(val, index, buf)
        return buf

def interval(starts, sizes, index, nleaves, leaf_ids):
    cdef unsigned char[:] c_index = index;
    cdef int[:] c_starts = starts;
    cdef int[:] c_sizes = sizes;
    cdef int c_nleaves = nleaves;
    cdef unsigned char[:] c_leaf_ids = leaf_ids;
    cdef int c_len = len(index)

    with nogil:
        find_interval(<int*>(&c_starts[0]), <int*>(&c_sizes[0]), <unsigned
                char*>(&c_index[0]), <int> c_len, <int> c_nleaves, <unsigned char*>(&c_leaf_ids[0]))


def bin_default(levels, real[:, :] projection, real[:] medians, index[:] output, real[:, :] buf):
    cdef size_t n = projection.shape[0]
    cdef int c_levels = levels

    bin_queries(n, c_levels, &projection[0, 0], &medians[0], &output[0], &buf[0, 0])
    
    
def bin_pack(levels, real[:, :] projection, real[:] medians, index[:] output, real[:, :] buf):
    cdef size_t n = projection.shape[0]
    cdef int c_levels = levels

    if (real is cython.float) and (index is cython.int):
        print("USING SIMD", flush=True)
        bin_queries_simd(n, c_levels, &projection[0, 0], &medians[0], &output[0], &buf[0, 0])
    else:
        bin_queries_pack(n, c_levels, &projection[0, 0], &medians[0], &output[0], &buf[0, 0])

def query_to_bin(projection, medians, output, levels=None, pack=False):

    if levels is None:
        levels = projection.shape[1]

    levels = min(projection.shape[1], levels)

    buf = np.empty((projection.shape[0], projection.shape[1]), dtype=projection.dtype)

    if pack:
        bin_pack(levels, projection, medians, output, buf)
    else:
        bin_default(levels, projection, medians, output, buf)

#-- Dense KNN

cpdef single_knn(gids, R, Q, k, cores):
    """
    Performs an exact search for the nearest neighbors of dense Q in dense R using the GSKNN kernel on the CPU. 

    Parameters:
        gids (1d-array, int32): labels of reference points
        R (2d-array, float32): Nr x d reference coordinates
        Q (2d-array, float32): Nq x d query coordinates
        k (int): number of nearest neighbors to find
        cores (int): set the number of openmp threads used.  

     Returns:
        neighbor_list (2d-array, int32): Nq x k list of neighbor ids
        neighbor_dist (2d-array, float32): Nq x k list of neighbor distances
    """

    cdef int cn = R.shape[0];
    cdef int cm = Q.shape[0];
    cdef int cd = Q.shape[1];
    cdef int ck = k;

    neighbor_list = np.zeros([cm, k], dtype=np.int32)
    neighbor_dist = np.zeros([cm, k], dtype=np.float32)

    cdef float[:, :] cR = R;
    cdef float[:, :] cQ = Q;
    cdef int[:, :] cNL = neighbor_list;
    cdef float[:, :] cND = neighbor_dist;
    cdef int[:] cgids = gids; 
    cdef int[:] cqids = np.arange(0, cm, dtype=np.int32);
    with nogil:
        GSKNN[float](&cgids[0], &cqids[0], &cR[0, 0], &cQ[0, 0], cn, cd, cm, ck, &cNL[0, 0], &cND[0, 0]);


    return neighbor_list, neighbor_dist 

cpdef batched_knn(gidsList, RList, QList, k, cores):
    cdef int nleaves = len(RList); #Assume input is proper #TODO: Add checks
    cdef int cd = RList[0].shape[1];
    cdef int ck = k;
    cdef int ccores = cores
    cdef size_t[:] cRList = np.zeros(nleaves, dtype=np.uintp);
    cdef size_t[:] cQList = np.zeros(nleaves, dtype=np.uintp);

    cdef int[:] cns = np.zeros(nleaves, dtype=np.int32);
    cdef int[:] cms = np.zeros(nleaves, dtype=np.int32);

    cdef size_t[:] cqgidsList = np.zeros(nleaves, dtype=np.uintp);
    cdef size_t[:] crgidsList = np.zeros(nleaves, dtype=np.uintp);

    cdef size_t[:] cNLList = np.zeros(nleaves, dtype=np.uintp);
    cdef size_t[:] cNDList = np.zeros(nleaves, dtype=np.uintp);


    #Define temporary variables
    
    cdef float[:, :] localR;
    cdef float[:, :] localQ;

    cdef int localn;
    cdef int localm;
   
    cdef int[:] qgids;
    cdef int[:] rgids;

    cdef int[:,:] cNL;
    cdef float[:,:] cND;

    NLL = []
    NDL = []
    #Make nlist, mlist
    for i in range(nleaves):
        localR = RList[i]; #argghhhh, these need the GIL. I can't do this in parallel
        localQ = QList[i];

        localn = localR.shape[0];
        localm = localQ.shape[0];
       
        rgids = gidsList[i]; #This needs the GIL
        qgids = np.arange(0, localm,dtype=np.int32);

        cNL = np.zeros([localm, ck], dtype=np.int32);
        cND = np.zeros([localm, ck], dtype=np.float32);

        NLL.append(np.asarray(cNL))
        NDL.append(np.asarray(cND))

        cns[i] = localn;
        cms[i] = localm;

        cRList[i] = <np.uintp_t>&localR[0, 0];
        cQList[i] = <np.uintp_t>&localQ[0, 0];

        crgidsList[i] = <np.uintp_t>&rgids[0];
        cqgidsList[i] = <np.uintp_t>&qgids[0];

        cNLList[i] = <np.uintp_t>&cNL[0, 0];
        cNDList[i] = <np.uintp_t>&cND[0, 0];

    with nogil:
        batchedGSKNN[float](<int**>(&crgidsList[0]), <int**>(&cqgidsList[0]), <float**>(&cRList[0]), <float**>(&cQList[0]), <int *>(&cns[0]), cd, <int*>(&cms[0]), ck, <int**>(&cNLList[0]), <float**>(&cNDList[0]), nleaves, <int> ccores);
    
    #NLL = np.asarray(cNLList);
    #NDL = np.asarray(cNDList);
    out = None
    return (NLL, NDL, out)

#-- Sparse KNN

cpdef sparse_exact(gids, R, Q, k, cores):
    """
    Performs an exact search of a sparse Q in a sparse R on the CPU. 
    Computes the full pairwise distance matrix and sorts. 

    Parameters:
        gids (1d-array, int32): labels of reference points
        R (CSR, int32, int32, float32): Nr x d reference coordinates
        Q (CSR, int32, int32, float32): Nq x d query coordinates
        k (int): number of nearest neighbors to find
        cores (int): set the number of openmp threads used.  

    Returns:
        neighbor_list (2d-array, int32): Nq x k list of neighbor ids
        neighbor_dist (2d-array, float32): Nq x k list of neighbor distances
    """

    nr, dr = R.shape
    nq, dq = Q.shape

    cdef int c_nr = nr
    cdef int c_dr = dr
    cdef int c_nq = nq
    cdef int c_dq = dq

    cdef int[:] hID = np.asarray(gids, dtype=np.int32);

    cdef float[:] datar = R.data
    cdef int[:] idxr = R.indices
    cdef int[:] ptrr = R.indptr

    cdef float[:] dataq = Q.data
    cdef int[:] idxq = Q.indices
    cdef int[:] ptrq = Q.indptr
    cdef unsigned int nnzr = R.nnz
    cdef unsigned int nnzq = Q.nnz

    cdef int c_k = k
    cdef int c_cores = cores

    cdef int[:, :] nID = np.zeros([nq, k], dtype=np.int32) + <int> 1
    cdef float[:, :] nDist = np.zeros([nq, k], dtype=np.float32) + <float> 1e38

    with nogil:
        exact_knn(c_nq, c_dq, nnzq, <int*> &ptrq[0], <int*> &idxq[0], <float*> &dataq[0], c_nr, c_dr, nnzr, <int*> &ptrr[0], <int*> &idxr[0], <float*> &datar[0], <int*> &hID[0], c_k, <int*> &nID[0, 0], <float*> &nDist[0, 0])

    outID = np.asarray(nID)
    outDist = np.asarray(nDist)

    return (outID, outDist)


cpdef sparse_knn_3(gids, pptr, pind, pval, pnnz, levels, ntrees, k, blocksize, cores, n, d):

    """
    Performs an approximate all-all search of a given CSR matrix on the CPU. 
    Forms a forest of randomized projection trees.  

    Parameters:
        gids (1d-array, int32): labels of reference points
        pptr (int32): CSR row ptr index 
        pind (int32): CSR col index 
        pval (float32): CSR values
        levels (int): # of levels in the projection tree 
        ntrees (int): # of tree iterations to perform 
        k (int): number of nearest neighbors to find
        blocksize (int): Number of points per leaf to search simultatenously
        cores (int): set the number of openmp threads used.
        n (int): number of data points in CSR
        d (int): dimension of CSR 

    Returns:
        neighbor_list (2d-array, int32): Nq x k list of neighbor ids
        neighbor_dist (2d-array, float32): Nq x k list of neighbor distances
    """


    cdef unsigned int[:] hID = np.asarray(gids, dtype=np.uint32);
    cdef float[:] data = pval
    cdef int[:] idx = pind
    cdef int[:] ptr = pptr
    cdef unsigned int nnz = pnnz

    cdef unsigned int c_n = n
    cdef unsigned int c_d = d

    cdef int c_k = k
    cdef int c_ntrees = ntrees
    cdef int c_blocksize = blocksize
    cdef int c_cores = cores    
    cdef int c_levels = levels

    cdef unsigned int[:, :] nID = np.zeros([n, k], dtype=np.uint32) + <unsigned int> 1
    cdef float[:, :] nDist = np.zeros([n, k], dtype=np.float32) + <float> 1e38

    with nogil:
        spknn(<unsigned int*> &hID[0], <int*> &ptr[0], <int*> &idx[0], <float*> &data[0], <unsigned int> c_n, <unsigned int> c_d, <unsigned int> nnz, <unsigned int*> &nID[0, 0], <float*> &nDist[0, 0], <int> c_k, <int> c_levels, <int> c_ntrees, <int> c_blocksize, <int> c_cores)

    outID = np.asarray(nID)
    outDist = np.asarray(nDist)

    return (outID, outDist)


cpdef sparse_knn(gids, X, levels, ntrees, k, blocksize, cores):
    """
    Performs an approximate all-all search of a given CSR matrix on the CPU. 
    Forms a forest of randomized projection trees.  

    Parameters:
        gids (1d-array, int32): labels of reference points
        X (CSR, int32, int32, float32): N x d coordinate CSR 
        levels (int): # of levels in the projection tree 
        ntrees (int): # of tree iterations to perform 
        k (int): number of nearest neighbors to find
        blocksize (int): Number of points per leaf to search simultatenously
        cores (int): set the number of openmp threads used.
        n (int): number of data points in CSR
        d (int): dimension of CSR 

    Returns:
        neighbor_list (2d-array, int32): Nq x k list of neighbor ids
        neighbor_dist (2d-array, float32): Nq x k list of neighbor distances
    """

    n, d = X.shape
    assert(n == len(gids))

    cdef unsigned int[:] hID = np.asarray(gids, dtype=np.uint32);
    cdef float[:] data = X.data
    cdef int[:] idx = X.indices
    cdef int[:] ptr = X.indptr
    cdef unsigned int nnz = X.nnz

    cdef unsigned int c_n = n
    cdef unsigned int c_d = d

    cdef int c_k = k
    cdef int c_ntrees = ntrees
    cdef int c_blocksize = blocksize
    cdef int c_cores = cores    
    cdef int c_levels = levels

    cdef unsigned int[:, :] nID = np.zeros([n, k], dtype=np.uint32) + <unsigned int> 1
    cdef float[:, :] nDist = np.zeros([n, k], dtype=np.float32) + <float> 1e38

    with nogil:
        spknn(<unsigned int*> &hID[0], <int*> &ptr[0], <int*> &idx[0], <float*> &data[0], <unsigned int> c_n, <unsigned int> c_d, <unsigned int> nnz, <unsigned int*> &nID[0, 0], <float*> &nDist[0, 0], <int> c_k, <int> c_levels, <int> c_ntrees, <int> c_blocksize, <int> c_cores)

    outID = np.asarray(nID)
    outDist = np.asarray(nDist)

    return (outID, outDist)

#-- Merge 

cpdef merge_neighbors(a, b, k, cores):
    merge_t = time.time()

    I1 = a[0]
    D1 = a[1]

    I2 = b[0]
    D2 = b[1]

    old_type = I1.dtype
    cdef float[:, :] cD1 = D1;
    cdef float[:, :] cD2 = D2;

    #print("before", I1)
    cdef unsigned int[:, :] cI1 = np.asarray(I1, dtype=np.uint32);
    cdef unsigned int[:, :] cI2 = np.asarray(I2, dtype=np.uint32);
    #print("after", np.asarray(cI1))

    cdef unsigned int cn = I1.shape[0]
    cdef int ck = k
    cdef int ccores = cores
    with nogil:
        merge_neighbor_cpu[float](&cD1[0, 0], &cI1[0, 0], &cD2[0, 0], &cI2[0, 0], cn, ck, <int> ccores)

    merge_t = time.time() - merge_t

    I1 = np.asarray(cI1, dtype=old_type)
    I2 = np.asarray(cI2, dtype=old_type)

    return (I1, D1)

#-- Local Tree Build  

"""
cpdef reorder(data, perm, n, d):
    cdef float[:, :] c_data = data 
    cdef int[:] c_perm = perm 
    cdef int c_n = n 
    cdef int c_d = n 

    with nogil:
        gather(&c_data[0, 0], &c_perm[0], c_n, c_d)
"""

cpdef dense_build(P):
    cdef unsigned int c_n = P.shape[0]
    cdef size_t c_L = P.shape[1]

    cdef float[:,:] c_P = P 
    #cdef float[:,:] c_data = data 

    cdef unsigned int[:] c_order = np.arange(c_n, dtype=np.uint32)
    cdef unsigned int[:] c_firstPt = np.empty(2**c_L, dtype=np.uint32)

    #Compute Tree Ordering
    with nogil:
        build_tree(&c_P[0,0], &c_order[0], &c_firstPt[0], c_n, c_L) 

    order = np.asarray(c_order)
    firstPt = np.asarray(c_firstPt)

    return (order, firstPt)

#-- Distributed Tree Build 

cpdef dist_select(int k, float[:] X, int[:] ID, comm, prev=(0, 0, 0)):

    rank = comm.Get_rank()
    cdef int nlocal = len(X)
    nlocal_a = np.array(nlocal, dtype=np.float32)
    N_a = np.array(0.0, dtype=np.float32)
    comm.Allreduce(nlocal_a, N_a, op=MPI.SUM)
   
    cdef float N = N_a

    if prev[2] == 0:
        globalN = N
    else:
        globalN = prev[2]

    if(k > globalN):
        raise Exception("Distributed Select: k cannot be greater than the total number of points.")
    
    #TODO: Combine this with previous MPI Call
    #Compute Mean and use as approximate Split
    if nlocal >0:
        local_mean_a = np.array(np.sum(X), dtype=np.float32)
    else:
        local_mean_a = np.array(0.0, dtype=np.float32)

    global_mean_a = np.array(0.0, dtype=np.float32)
    comm.Allreduce(local_mean_a, global_mean_a, op=MPI.SUM)
    #print(global_mean_a)
    cdef float mean = global_mean_a/N
    #print(rank, "Mean", mean)

    #TODO: Replace with parallel scan and copy
    cdef float[:] temp_X = np.zeros(nlocal, dtype=np.float32)
    cdef int[:] temp_ID = np.zeros(nlocal, dtype=np.int32)
    cdef int[:] perm = np.zeros(nlocal, dtype=np.int32)

    cdef int nleft = 0;
    cdef int nright = 0;
    cdef float current_value

    cdef int i = 0
    for i in range(nlocal):
        current_value = X[i]
        if( current_value <= mean):
            #print(rank, current_value)
            perm[i] = nleft
            nleft = nleft + 1
        else:
            perm[i] = nlocal - 1 - nright
            nright = nright + 1
    i = 0
    for i in prange(nlocal, nogil=True):
        temp_X[perm[i]] = X[i]
        temp_ID[perm[i]] = ID[i]

    i = 0
    for i in prange(nlocal, nogil=True):
        X[i] = temp_X[i]
        ID[i] = temp_ID[i] 

    cdef int nL = prev[0] + nleft;
    cdef int nR = prev[1] + nright;

    cdef int[:] local_split_info = np.array([nL, nR], dtype=np.int32)
    cdef int[:] global_split_info = np.array([0, 0], dtype=np.int32)
    comm.Allreduce(local_split_info, global_split_info, op=MPI.SUM)

    if nlocal > 0:
        minX = np.min(X)
        maxX = np.max(X)
    else:
        minX = 3.4e38
        maxX = -3.4e38

    gmax = comm.allreduce(maxX, op=MPI.MAX)
    gmin = comm.allreduce(minX, op=MPI.MIN)

    cdef int global_nleft = global_split_info[0]
    cdef int global_nright = global_split_info[1]

    if (math.isclose(gmax, gmin, rel_tol=1e-5, abs_tol=1e-20)):
        st0 = np.random.get_state()
        np.random.seed(None)
        #print("Warning: Up to precision ", N, " points are the same.")
        X = X + np.array(np.random.randn(nlocal), dtype=np.float32)*gmax*(3e-4)
        np.random.set_state(st0)

    if (global_nleft == k) or (N == 1) or (global_nleft == globalN) or (global_nright == globalN):
        return (mean, nL)

    elif (global_nleft > k):
        return dist_select(k, X[:nleft], ID[:nleft], comm, prev=(prev[0], prev[1] + nright, globalN))

    elif (global_nright > k):
        return dist_select(k, X[nleft:], ID[nleft:], comm, prev=(prev[0]+nleft, prev[1], globalN))
