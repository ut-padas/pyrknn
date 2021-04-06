import numpy as np
import cupy as cp
from numba import njit, jit, prange, cuda



def l2(q,r):
    qnr = cp.sum(q**2,axis=-1)
    rnr = cp.sum(r**2,axis=-1)
    d = -2*q@r.T
    n,m=d.shape
    qm = cp.tile(qnr,(m,1)); 
    qm = qm.T
    d += qm + cp.tile(rnr,(n,1))
    return cp.sqrt(cp.abs(d)) 

def orthoproj(X,numdir):
    n,dim = X.shape
    U = cp.random.randn(dim,numdir)
    Q,_ =cp.linalg.qr(U,mode='reduced')
    U[:,:Q.shape[1]] = Q
    Xr = X.dot(U)
    return (Xr,U)

def segpermute(arr,segsize,gperm):
    n = len(arr)
    offsets = cp.arange(0,n,segsize)
    for i in range(len(offsets)):
        st = offsets[i]
        en = min(st+segsize, n)
        perm = cp.argsort(arr[st:en])
        gperm[st:en] = perm+st

def segpermute_f(arr,segsize,perm):
    n = len(arr)
    b = n//segsize
    perm[:] = cp.argsort(arr.reshape(b,segsize), axis =1).flatten()
    offsets = cp.arange(0,n,segsize)
    st = cp.tile(offsets, (segsize,1))
    st = (st.T).ravel()
    perm[:] = perm[:] + st[:]
    
    

def merge_knn(dis,idx,k):
    m = len(dis)
    tmp_idx = cp.empty_like(idx)
    for j in range(m):
        _,tmp_idx = cp.unique(idx[j,],return_index=True)
        m = len(tmp_idx)
        dis[j,:m] = dis[j,tmp_idx]
        idx[j,:m] = idx[j,tmp_idx]        
        tmp_idx = cp.argsort(dis[j,:m])
        dis[j,:k]=dis[j,tmp_idx[:k]]
        idx[j,:k]=idx[j,tmp_idx[:k]]
