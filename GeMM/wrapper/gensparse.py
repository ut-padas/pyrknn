import numpy as np
from scipy.sparse import csr_matrix
from cuda_wrapper.core import py_gpuknn
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
import cupy as cp

def gen_random_sparse_csr(N,M,avg_nnz,idtype=np.int32,vltype=np.float32):
    '''
    Generate a real random sparse matrix of N-by-M dimensions with avz_nnz
    nonzeros per row:  total memory ~ N*avz_nnz
    The values are normally distributed
    The nonzeros are uniformly distributed
    
    Does not allow empty rows, (at least )
    
    
    Parameters
    ----------
    N : int
        number of rows
    M : int
        number of columns (full matrix)
    avg_nnz : int
        average number of nonzeros _per_row_
    idtype : type, optional
        type for indeces The default is np.int32.
    vltype : type, optional
        type for matrix values. The default is np.float32.

    Returns
    -------
    X : sparse N-by-M  matrix
        CSR format.
        
    Example:
    N=10
    avg_nnz_per_row = 4
    M = 9
    idtype = np.int32
    vltype = np.float32

    X = gen_random_sparse_csr(N,M,avg_nnz_per_row)
    print(X.toarray())


    '''

    # generate random connectivity matrix
    con = np.random.randint(1,M+1,size=(N,2*avg_nnz), dtype=idtype)
    con = unique_byrow(con)  # detect duplicates
    con = np.sort(con,axis=1)  # sort indices
    flt = np.random.randint(2,size=(N,2*avg_nnz))
    flt = np.sort(flt,axis=1)
    con = con*flt
    cols = con.ravel()  # get column
    cols = cols[cols>0] 
    nnz_arr = np.sum(con>0,axis=1) # nonzeros per rows

    nnz = np.sum(nnz_arr)        
    rows = np.block( [0, np.cumsum(nnz_arr)])
    cols -=1
    vals = np.random.randn(nnz).astype(vltype)
    X =csr_matrix((vals, cols, rows), shape=(N,M) )
    return X


def unique_byrow(a):
    weight = 1j*np.linspace(0, a.shape[1], a.shape[0], endpoint=False)
    b = a + weight[:, np.newaxis]
    u, ind = np.unique(b, return_index=True)
    b = np.zeros_like(a)
    np.put(b, ind, a.flat[ind])
    return b


if __name__ == '__main__':

    LogNP = 21
    N = 1 << LogNP	# number of points
    LogPPL = 10			# number of points per leaf
    depth = max(0, LogNP - LogPPL)
    points_per_leaf = 1 << (LogNP - depth)
     
    avg_nnz_per_row = 16
    M = 1000
    idtype = np.int32
    vltype = np.float32
    X = gen_random_sparse_csr(N,M,avg_nnz_per_row)
    gids = np.random.permutation(np.arange(N))
    leaves = 1 << depth 
    K = 32
    knnidx = np.random.randint(N, size=(N,K),dtype=np.int32)
    knndis = 1e30*np.ones((N,K),dtype=np.float32)

    print(f'Total nnz = {X.nnz}')
    nnzarr = np.diff(X.indptr)
    maxnnz = np.ceil(np.log2(np.max(nnzarr)))
    print(f'Avg nnz per row = {np.mean(nnzarr)}')
    print(f'Max nnz per row = {np.max(nnzarr)}')
    print(f'Min nnz per row = {np.min(nnzarr)}')
     
    py_gpuknn(X.indptr, X.indices, X.data, gids, \
                             N, leaves, \
                             K, knndis.ravel(),knnidx.ravel(), \
                             maxnnz)


    if 1: # check accuracy for first leaf
        h_knndis = cp.asnumpy(knndis)
        h_knnidx = cp.asnumpy(knnidx)
				
        data    = cp.asnumpy(X.data)
        indptr  = cp.asnumpy(X.indptr)
        indices = cp.asnumpy(X.indices)
        hX = csr_matrix((data,indices,indptr), shape=(N,dim))
        # compute exact knns using sklearn
        nex = points_per_leaf
        t = 0
        nbrs = NearestNeighbors(n_neighbors=K,algorithm='brute').fit(hX[t*nex:(t+1)*nex,:])
        knndis_ex, knnidx_ex = nbrs.kneighbors(hX[t*nex:(t+1)*nex,:])
        print('true')
        #print(knndis_ex)
        #print(knnidx_ex)
        print('rec')
        #print(h_knndis[t*nex:(t+1)*nex, :])
        #print(h_knnidx[t*nex:(t+1)*nex, :])
        #print(h_knnidx)
        
        ex = knnidx_ex
        ap = h_knnidx
        rowerr = np.any(ex[:nex,]-ap[t*nex:(t+1)*nex,],axis=1)
        rowidx = np.where(rowerr==True)
        acc = 1 - len(rowidx[0])/nex
        print('Recall accuracy:', '{:.4f}'.format(acc))








