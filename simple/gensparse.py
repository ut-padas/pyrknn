import numpy as np
from scipy.sparse import csr_matrix
USE_CUDA=True


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
    
    nnz_arr = np.random.randint(2, 2*avg_nnz, size=N,dtype=idtype)
    nnz = np.sum(nnz_arr)
    cols = np.random.randint(M, size=nnz, dtype=idtype)
    rows = np.block( [0, np.cumsum(nnz_arr)])
    vals = np.random.randn(nnz).astype(vltype)
    X =csr_matrix((vals, cols, rows), shape=(N,M) )
    X.sort_indices()    
    return X



# N=10
# avg_nnz_per_row = 4
# M = 9
# idtype = np.int32
# vltype = np.float32

# X = gen_random_sparse_csr(N,M,avg_nnz_per_row)
# print(X.toarray())


if USE_CUDA:
    import cupy as cp

    def gen_random_sparse_csr_gpu(N,M,avg_nnz,idtype=cp.int32,vltype=cp.float32):
        nnz_arr = cp.random.randint(2, 2*avg_nnz, size=N, dtype=idtype)
        nnz = cp.sum(nnz_arr)
        cols = cp.random.randint(M, size=int(nnz), dtype=idtype)
        rows = cp.concatenate( [cp.zeros(1), cp.cumsum(nnz_arr)])
        vals = cp.random.randn(int(nnz)).astype(vltype)
        X = cp.sparse.csr_matrix((vals,cols,rows), shape=(N,M))
        X.sort_indices()
        return X
        
