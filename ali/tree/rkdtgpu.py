
import tree.utilsgpu as ut
from time import time
import cupyx.scipy.sparse as cpsp
import sys
import time
import cupy as cp
import matplotlib.pyplot as plt


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


        
def rkdt_a2a_it(Xref, Xq, levels, knnidx, knndis, K, maxit,monitor=None, overlap=0, dense=True, deviceId=0, batchIter=10):
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
    n = Xref.shape[0]
    nq = Xq.shape[0]
    leaves = 1 << levels
    
     
    mempool = cp.get_default_memory_pool()
    numiters = cp.ceil(maxit/batchIter)
    for t in cp.arange(numiters):
      
      morId_q = cp.zeros((nq, batchIter), dtype = cp.int32) 
      gids_batches = cp.zeros((n, batchIter), dtype = cp.int32)

      tic = time.time()
      for b_t in cp.arange(batchIter):
       
        gids_batches[:, b_t] = cp.arange(0, n,dtype=cp.int32)
        perm = cp.arange(0, n,dtype=cp.int32)
        Pref, Pq = ut.orthoproj_query(Xref, Xq, levels)
        segsize = n
        for i in range(0,levels):
            
          segsize = n>>i
          perm , morId_q[:, b_t] = ut.segpermute_f_query(Pref[:,i], Pq[:,i], segsize,perm, morId_q[:, b_t])    
          Pref[:,i]=Pref[:,i][perm]
           
          gids_batches[:, b_t]=gids_batches[:, b_t][perm]
        del Pref
        del Pq
      
      toc = time.time() - tic
       
      '''
      print("batch of %d, Tree construction takes %.4f sec"%(batchIter,toc))
      mempool = cp.get_default_memory_pool()
      mempool.free_all_blocks() 
      if dense:
          dim = X.shape[1]
          py_dfiknn(gids_batches, Xref, Xq, leaves, K, knnidx, knndis, dim, deviceId, morId_q, batchIter) 
      if not dense:
          py_sfiknn(gids_batches, Xref, Xq, leaves, K, knndis, knnidx, deviceId, morId_q, batchIter) 
      begin_err = time.time()
      if monitor is not None:
          if monitor(t,knnidx,knndis):
              break
      end_err = time.time()
      err_time = end_err - begin_err
      tot_err += err_time
      cur_time = time.time() - tot_err - begin
      print("it = %d, RKDT : %.4f sec"%(t, cur_time))
      '''
    end = time.time()
    tot_rkdt = end - begin - tot_err
    print("RKDT takes %.4f sec "%tot_rkdt)
    print("Error takes %.4f sec "%tot_err)

    return knnidx, knndis






 
    
