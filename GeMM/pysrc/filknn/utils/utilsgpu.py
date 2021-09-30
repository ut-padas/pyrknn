#import numpy as np
import cupy as cp
import cupyx.scipy.sparse as cpsp

def spnrm(X,axis):
    Y=cp.sparse.csr_matrix((X.data**2,X.indices,X.indptr))
    return cp.sqrt(Y.sum(axis))

def l2sparse(q,r):
    #rownormq = cpsp.linalg.norm if cpsp.issparse(q) else cp.linalg.norm
    #rownormr = cpsp.linalg.norm if cpsp.issparse(r) else cp.linalg.norm
    rownormq = spnrm if cpsp.issparse(q) else cp.linalg.norm
    rownormr = spnrm if cpsp.issparse(r) else cp.linalg.norm    
    qnr = rownormq(q,axis=1)**2
    rnr = rownormr(r,axis=1)**2
    d = -2*q@r.T
    d = d.toarray()
    n,m= d.shape
    qm = cp.tile(qnr.ravel(),(m,1)); 
    qm = qm.T
    d += cp.squeeze(cp.asarray(qm + cp.tile(rnr.ravel(),(n,1))))
    return cp.sqrt(cp.abs(d)) 


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
    #Xr = cp.dot(X,U)
    #Xr = cp.asarray(Xr)
    del Q
    
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
    perm[:] = cp.argsort(arr.reshape(b,segsize), axis = 1).flatten()
    offsets = cp.arange(0,n,segsize)
    st = cp.tile(offsets, (segsize,1))
    st = (st.T).ravel()
    perm[:] = perm[:] + st[:]
    del offsets
    del st
    return perm 

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


# THIS works only for contiguous local sets (ls) and it won't work for random sliced views
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
    return Z

    if 0:
        A=X.toarray()
        B=Z.toarray()
        print(A[ls,]-B)

