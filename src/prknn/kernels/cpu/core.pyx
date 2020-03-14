# distutils: language = c++
#cython: boundscheck=False
import numpy as np
cimport numpy as np

from primitives cimport *
from libcpp.vector cimport vector
from cython.operator cimport dereference as deref
from cython.parallel import prange, parallel

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

cpdef PyGSKNN(gids, R, Q, k):
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

cpdef PyGSKNNBatched(gidsList, RList, QList, k):
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
        batchedGSKNN[float](<int**>(&crgidsList[0]), <int**>(&cqgidsList[0]), <float**>(&cRList[0]), <float**>(&cQList[0]), <int *>(&cns[0]), cd, <int*>(&cms[0]), ck, <int**>(&cNLList[0]), <float**>(&cNDList[0]), nleaves);
    
    #NLL = np.asarray(cNLList);
    #NDL = np.asarray(cNDList);
    return (NLL, NDL)


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
