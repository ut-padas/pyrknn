# distutils: language = c++
#cython: boundscheck=False

import numpy as np
cimport numpy as np

from primitives cimport *
from libcpp.vector cimport vector
from cython.operator cimport dereference as deref
from cython.parallel import prange, parallel
from cython.view cimport array as cvarray
import cython

cimport mpi4py.MPI as MPI
cimport mpi4py.libmpi as libmpi
from mpi4py import MPI
#import time as MPI

import time

cpdef sparse_exact(gids, R, Q, k, cores):
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



cpdef sparse_knn(gids, X, levels, ntrees, k, blocksize, cores):
    n, d = X.shape
    assert(n == len(gids))

    cdef unsigned int[:] hID = gids;
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

cpdef KNNLowMem(gids, R, Q, k):
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
    with nogil:
        directKLowMem[float](&cgids[0], &cR[0, 0], &cQ[0, 0], cn, cd, cm, ck, &cNL[0, 0], &cND[0, 0]);

    return neighbor_list, neighbor_dist

cpdef single_knn(gids, R, Q, k, cores):
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

cpdef PyGSKNNBlocked(gids, R, Q, k):
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
        blockedGSKNN[float](&cgids[0], &cqids[0], &cR[0, 0], &cQ[0, 0], cn, cd, cm, ck, &cNL[0, 0], &cND[0, 0]);

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


cpdef RefBatched(gidsList, RList, QList, k):
    cdef int nleaves = len(RList); #Assume input is proper #TODO: Add checks
    cdef int cd = RList[0].shape[1];
    cdef int ck = k;

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
        batchedRef[float](<int**>(&crgidsList[0]), <int**>(&cqgidsList[0]), <float**>(&cRList[0]), <float**>(&cQList[0]), <int *>(&cns[0]), cd, <int*>(&cms[0]), ck, <int**>(&cNLList[0]), <float**>(&cNDList[0]), nleaves);
    
    #NLL = np.asarray(cNLList);
    #NDL = np.asarray(cNDList);
    return (NLL, NDL)

cpdef test_par():
    with nogil:
        test[float]();

cpdef choice(arr):
    cdef int N = len(arr);
    cdef int[:] anchors = np.zeros(2, dtype=np.int32)
    with nogil:
        choose2(&anchors[0], N)
    return anchors

cpdef kselect(arr, k):
    cdef int c_k = k
    cdef int N = len(arr)
    cdef float[:] c_arr = arr;
    print("Entered Select")
    with nogil:
        quickselect[float](&c_arr[0], N, c_k)



cpdef merge_neighbors(a, b, k, cores):
    merge_t = time.time()

    I1 = a[0]
    D1 = a[1]

    I2 = b[0]
    D2 = b[1]

    cdef float[:, :] cD1 = D1;
    cdef float[:, :] cD2 = D2;

    cdef int[:, :] cI1 = I1;
    cdef int[:, :] cI2 = I2;

    cdef int cn = I1.shape[0]
    cdef int ck = k
    cdef int ccores = cores
    with nogil:
        merge_neighbor_cpu[float](&cD1[0, 0], &cI1[0, 0], &cD2[0, 0], &cI2[0, 0], cn, ck, <int> ccores)

    merge_t = time.time() - merge_t
    print("Merge time:", merge_t)

    return (I1, D1)



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

    #print(rank, "prev", prev)
    #print(rank, "local_n, global_n", (nlocal, N))
    #print(rank, "local_local_splits", [nleft, nright])
    #print(rank, "local_splits", np.asarray(local_split_info))
    #print(rank, "local total", np.sum(local_split_info))
    #print(rank, "Global split", np.asarray(global_split_info))
    #print(rank, "global total", np.sum(global_split_info))
    #print(rank, "k", k)
    #print(rank, "mean", mean)
    #print(rank, "X", np.array(X))

    if nlocal > 0:
        minX = np.min(X)
        maxX = np.max(X)
    else:
        minX = 3.4e38
        maxX = -3.4e38

    gmax = comm.allreduce(maxX, op=MPI.MAX)
    gmin = comm.allreduce(minX, op=MPI.MIN)
    #print(rank, "minX", gmin)
    #print(rank, "maxX", gmax)

    cdef int global_nleft = global_split_info[0]
    cdef int global_nright = global_split_info[1]

    if (gmax - gmin < 0.00001):
        X = X + np.array(np.random.rand(nlocal), dtype=np.float32)*(1e-3)

    if (global_nleft == k) or (N == 1) or (global_nleft == globalN) or (global_nright == globalN):
        #print(rank, "Mean", mean)
        #print(rank, "NLEFT", nL)
        return (mean, nL)

    elif (global_nleft > k):
        #print(rank, "left")
        return dist_select(k, X[:nleft], ID[:nleft], comm, prev=(prev[0], prev[1] + nright, globalN))

    elif (global_nright > k):
        #print(rank, "right")
        return dist_select(k, X[nleft:], ID[nleft:], comm, prev=(prev[0]+nleft, prev[1], globalN))

def scan(l):
    if type(l) is not np.ndarray:
        raise TypeError("The passed argument is not a numpy array.")
    if l.dtype == np.dtype('int32'):
        return Scan[int,int](l)
    elif l.dtype == np.dtype('float32'):
        return Scan[float,float](l)
    elif l.dtype == np.dtype('float64'):
        return Scan[double,double](l)
    else:
        raise TypeError("Unsupported data element type.")

def sample_without_replacement(l,int n):
    if type(l) is not np.ndarray:
        raise TypeError("The passed argument is not a numpy array.")
    if l.dtype == np.dtype('int32'):
        return sampleWithoutReplacement[int](n,l)
    elif l.dtype == np.dtype('float32'):
        return sampleWithoutReplacement[float](n,l)
    elif l.dtype == np.dtype('float64'):
        return sampleWithoutReplacement[double](n,l)

def accumulate(l, i):
    if type(l) is not np.ndarray:
        raise TypeError("The passed argument is not a numpy array.")
    if l.dtype == np.dtype('int32'):
        return Accumulate[int](l,i)
    elif l.dtype == np.dtype('float32'):
        return Accumulate[float](l,i)
    elif l.dtype == np.dtype('float64'):
        return Accumulate[double](l,i)

def reduce_par(l, i):
    if type(l) is not np.ndarray:
        raise TypeError("The passed argument is not a numpy array.")
    if l.dtype == np.dtype('int32'):
        return Reduce[int](l,i)
    elif l.dtype == np.dtype('float32'):
        return Reduce[float](l,i)
    elif l.dtype == np.dtype('float64'):
        return Reduce[double](l,i)


def select(l, size_t k):
    if type(l) is not np.ndarray:
        raise TypeError("The passed argument is not a numpy array.")
    if l.dtype == np.dtype('int32'):
        return Select[int](k,l)
    elif l.dtype == np.dtype('float32'):
        return Select[float](k,l)
    elif l.dtype == np.dtype('float64'):
        return Select[double](k,l)
