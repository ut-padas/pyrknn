import numpy as np
from numba import njit
import scipy as sp
from scipy.sparse import csr_matrix, isspmatrix



def centerpoints(X):
    n=len(X)
    C=np.sum(X,axis=0)/n;
    X=X-C

@njit
def rank2_update(A,v,w):
    ''' 
    just adds a rank 2 update on a matrix
    A += v*e' + e*w'
    '''
    n = len(v) if v.shape else 1
    m = len(w)
    if n == 1 :
        A = v + w
        return
    
    for i in range(n):
        for j in range(m):
            A[i,j]+=v[i] + w[j]


def l2(q,r):
    """
    return Eucledian distance d = ||q-r||^2_2
    q: query points      2D array:  number-of-points BY dimnens
    r: reference points  2D array:  number-of-points BY dimnens
    """
    qnr = np.sum(q**2,axis=-1)
    rnr = np.sum(r**2,axis=-1)
    d = -2*q@r.T
    print(d)
    rank2_update(d,qnr,rnr)
    return np.sqrt(np.abs(d))   # abs to avoid issues with cancellation error in subtraction

def l2sparse(q,r):
    rownormq = sp.sparse.linalg.norm if sp.sparse.issparse(q) else np.linalg.norm
    rownormr = sp.sparse.linalg.norm if sp.sparse.issparse(r) else np.linalg.norm
    qnr = rownormq(q,axis=1)**2
    rnr = rownormr(r,axis=1)**2
    d = -2*q@r.T
    n,m= d.shape
    qm = np.tile(qnr,(m,1)); 
    qm = qm.T
    d += qm + np.tile(rnr,(n,1))
    d = np.squeeze(np.asarray(d))
    return np.sqrt(np.abs(d)) 


def orthoproj(X, numdir):
    """
    Input
    --------
    X: input points  n-by-dim,  n is number of points
    numdir: number of directions

    Output
    -----------
    Xpr - projections
    projdirs  - dim-by-dir, projection vectors and 
    """
    n,dim = X.shape
    use_random_pca = False

    if not use_random_pca:
        U = np.random.randn(dim,numdir)
        if 0: print(U.shape)
    else:
        prm = np.random.permutation(n)
        Xs = X[prm[:2*dim],:]
        U = Xs.T@Xs
    
    if not isspmatrix(X):
        Q,_ =np.linalg.qr(U,mode='reduced')
        U[:,:Q.shape[1]] = Q

    Xr = X@U
    return (Xr, U)


def segpermute(arr,segsize,gperm):
    '''
    Computes segmented permutation of the array 'arr'
      and stores it in 'gperm'
    'segsize' is size of each segment
    supports any size arrays
    '''
    n = len(arr)
    offsets = np.arange(0,n,segsize)
    for i in range(len(offsets)):
        st = offsets[i]
        en = min(st+segsize, n)
        perm = np.argsort(arr[st:en])
        gperm[st:en] = perm+st

def segpermute_f(arr,segsize,perm):
    '''
    Computes segmented permutation of the array 'arr'
      and stores it in 'gperm'
    'segsize' is size of each segment
    supports only n\\segment arrays (easy to fix with padding)
    '''

    n = len(arr)
    b = n//segsize
    perm[:] = np.argsort(arr.reshape(b,segsize), axis =1).flatten()
    offsets = np.arange(0,n,segsize)
    st = np.tile(offsets, (segsize,1))
    st = (st.T).ravel()
    perm[:] = perm[:] + st[:]
    
    


def merge_knn(dis,idx,k):
    '''
    Merges nearest neighbor and their global indeces
    dis - distances
    idx - global indices
    k nearest neighbors

    dis and idx have distances in any order
    duplicates allowed
    upon exit
    dis : has the 'k' smallest unique distances
    idx : has the corresponding values

    Notice that if no-k unique distances exist, then  output
    will contain duplicates.
    
    '''
    m = len(dis)
    tmp_idx = np.empty_like(idx)
    for j in range(m):
        _,tmp_idx0 = np.unique(idx[j,],return_index=True)
        m = len(tmp_idx0)
        dis[j,:m] = dis[j,tmp_idx0]
        idx[j,:m] = idx[j,tmp_idx0]        
        tmp_idx = np.argsort(dis[j,:m])
        dis[j,:k]=dis[j,tmp_idx[:k]]
        idx[j,:k]=idx[j,tmp_idx[:k]]
    






    
    
    
