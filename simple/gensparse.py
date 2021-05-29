import numpy as np
from scipy.sparse import csr_matrix

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

    N=10000
    avg_nnz_per_row = 100
    M = 1000
    idtype = np.int32
    vltype = np.float32
    X = gen_random_sparse_csr(N,M,avg_nnz_per_row)

    print(f'Total nnz = {X.nnz}')
    nnzarr = np.diff(X.indptr)
    print(f'Avg nnz per row = {np.mean(nnzarr)}')
    print(f'Max nnz per row = {np.max(nnzarr)}')
    print(f'Min nnz per row = {np.min(nnzarr)}')
    
