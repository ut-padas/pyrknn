import numpy as np

import utilsgpu as ut
from time import time
import cupyx.scipy.sparse as cpsp
import sys
sys.path.append("../")
import time
from sparse.sparse import *
from dense.dense import *


#@jit        
# TODO: THIS NEEDS TO BE REPLACED BY THE NEW CODE
def leaf_knn(X,gids,m,knnidx,knndis,k,init,overlap=0):
    '''
    Parameters
    ----------
    X : float matrix
        points: N - by -dim;  X[i,j] is the jth coordinate of the ith point
    gids : int
        global indices of points 
    m : int
        number of points per leaf
        X[:m,] is the first leaf, X[m:2m,] is the second leaf etc..
    knnidx : int matrix
        knnidx[i,j] is jth neighbor of ith point
    knndis : float matrix
        knndis[i,j] is the ith's point  jth neighbor distance to the ith point
    
    k : int
        number of nearest neighbors
    init : boolean
        True if this is the first tree iteration
    overlap : int
        Spill tree width parameter. The default is 0.

    Returns
    -------
    updates knnidx and knndis
    '''
    
    n = gids.shape[0]
    offsets = cp.arange(0,n,m)
    for i in range(len(offsets)):
        st = cp.asnumpy(offsets[i])
        en = min(st+m, n)
        ls =  gids[st:en]    # leaf set
        ov=overlap
        lss = gids[max(st-ov,0):min(n,en+ov)]
        D = ut.l2sparse(X[ls,:],X[lss,:])
        T=cp.tile(lss,(en-st,1))
        S = cp.argsort(D,axis=1)
        T=cp.take_along_axis(T,S,axis=1)
        D=cp.take_along_axis(D,S,axis=1)
        if init:
            knnidx[ls,:]=T[:,:k]
            knndis[ls,:]=D[:,:k]
 
        kit=cp.concatenate( (knnidx[ls,:],T[:,:k]), axis = 1)
        kdt=cp.concatenate( (knndis[ls,:],D[:,:k]), axis = 1)
        ut.merge_knn(kdt,kit,k)
        knndis[ls,:]=kdt[:,:k]
        knnidx[ls,:]=kit[:,:k]


        
def rkdt_a2a_it(X,levels,knnidx,knndis,K,maxit,monitor=None,overlap=0,dense=True, deviceId=0):
    '''
    Parameters
    ----------
    X : float matrix
        Logicaly X[i,j] is the jth coordinate of the ith point. Can be dense or sparse
    gids : int array
        global ids of points
    levels : int
        Tree depth
    knnidx : int matrix
        nearest neighbor indices
    knndis : float matrix
        nearest neighbor floats
    K : int
        number of nearest neighbors
    maxit : int
        maximum number of tree iterations
    monitor : function, optional
        allows early termination. The default is None.
    overlap : int, optional
        see leaf_knn() The default is 0.

    Returns
    -------
    knnidx : int matrix
        see above
    knndis : float matrix
        See above
    '''
    begin = time.time()
    cp.cuda.Device(deviceId).use()
    tot_err = 0.0
    tot_rkdt = 0.0
    n = X.shape[0]
    
    #perm = cp.empty_like(gids)
    #perm = cp.arange(n, dtype = cp.int32)
    
    mempool = cp.get_default_memory_pool()
    for t in range(maxit):
        
        tic = time.time()
        #gids = np.arange(0, n,dtype=np.int32)
        gids = cp.arange(0,n,dtype=cp.int32)
        perm = cp.arange(0, n,dtype=cp.int32)
        P = ut.orthoproj(X,levels)
        segsize = n
        for i in range(0,levels):
            
            segsize = n>>i
            perm = ut.segpermute_f(P[:,i],segsize,perm)    
            P[:,:]=P[perm,:]
            
            gids[:]=gids[perm]

            if 0: print(gids)
        leaves = 1 << levels
        toc = time.time() -tic
        del P
        del perm
        print("Tree construction takes %.4f sec"%toc)
        mempool = cp.get_default_memory_pool()
        mempool.free_all_blocks() 
        #print("befcuda %.4f free from %.4f "%(mempool.used_bytes()/1e9, mempool.total_bytes()/1e9))
        if dense:
            dim = X.shape[1]
            py_dfiknn(gids, X, leaves, K, knnidx, knndis, dim, deviceId) 
        if not dense:
            #print("\t Sparse knn : sfiknn version")
            py_sfiknn(gids, X, leaves, K, knndis, knnidx, deviceId) 
        begin_err = time.time()
        if monitor is not None:
            if monitor(t,knnidx,knndis):
                break
        end_err = time.time()
        err_time = end_err - begin_err
        tot_err += err_time
        cur_time = time.time() - tot_err - begin
        print("it = %d, RKDT : %.4f sec"%(t, cur_time))
    end = time.time()
    tot_rkdt = end - begin - tot_err
    print("RKDT takes %.4f sec "%tot_rkdt)
    print("Error takes %.4f sec "%tot_err)

    return knnidx, knndis






 
    
