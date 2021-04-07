import numpy as np
import cupy as cp
import utilsgpu as ut
from time import time
import cupyx.scipy.sparse as cpsp

def sliced_view(X,ls):
    if not cpsp.issparse(X):
        return X[ls,]
        
    rowst = ls[0]
    rowen = rowst+ls.shape[0]
    colst = X.indptr[rowst]
    colen = X.indptr[rowen]

    rptr = cp.copy(X.indptr[rowst:rowen+1])
    cptr = X.indices[colst:colen]
    vptr = X.data[colst:colen]
    rptr -= rptr[0]

    Z = cpsp.csr_matrix((vptr,cptr,rptr), shape=(ls.shape[0],X.shape[1]))

    if 0:
        A=X.toarray()
        B=Z.toarray()
        print(A[ls,]-B)


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

        D = ut.l2sparse( sliced_view(X,ls),  sliced_view(X,lss) )

        #D = ut.l2sparse(X[ls,:],X[lss,:])
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


        
def rkdt_a2a_it(X,gids,levels,knnidx,knndis,K,maxit,monitor=None,overlap=0,dense=True):
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
    n = X.shape[0]
    perm = cp.empty_like(gids)
    for t in range(maxit):
        gids[:] = cp.arange(0,n,dtype=cp.int32)
        perm[:] = cp.arange(0,n,dtype=cp.int32)
        P,_ = ut.orthoproj(X,levels)
        segsize = n
        for i in range(0,levels):
            segsize = n>>i
            ut.segpermute_f(P[:,i],segsize,perm)
            P[:,:]=P[perm,:]
            gids[:]=gids[perm]
            if 0: print(gids)


        if 1 and dense:
            leaf_knn(X,gids,segsize,knnidx,knndis,K,t==0,overlap)

        if monitor is not None:
            if monitor(t,knnidx,knndis):
                break
        
        if not dense:
            print("\t _WARNING_ :Sparse knn search not supported")









    
    
    
