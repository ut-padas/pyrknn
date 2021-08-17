import numpy as np
import cupy as cp
from time import time
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

import sys
#sys.path.append("../SpGeMM")
#import SpGeMM_2D_NoPerm as sgemm
from cuda_wrapper.sparse import *


if 1: 
    cp.random.seed(1)

LogNP =21; 
n=1<<LogNP   # NUMBER of POINTS

LogPPL=10
depth = max(0,LogNP-LogPPL)
points_per_leaf = 1<< (LogNP-depth)  # POINTS per leaf


print('Generate sparse array')
dim = 100
avg_nnz = 16
X = cp.sparse.random(n,dim, density = avg_nnz/dim, format='csr', dtype=cp.float32)

X.sort_indices()

print('Generate gids')
#gids = cp.random.permutation(cp.arange(n))
gids = cp.arange(n)

print('Generate knn arrays')
K=32
knnidx = cp.random.randint(n, size=(n,K),dtype=cp.int32)
knndis = 0.0*cp.ones((n,K),dtype=cp.float32)


#knndis[:,0] =0.0
#knnidx[:,0] = cp.arange(n)
maxnnz = cp.max(cp.diff(X.indptr))
leaves = 1<<depth

print('Entering SpGEMM')
print('%d total points, %d leaves, %d neighbors'%(n,leaves,K))
print('points_per_leaf =', points_per_leaf)

numits = 1



for k in range(1):
    tic = time()
    if 1:
        indptr_np = cp.asnumpy(X.indptr)
        indices_np = cp.asnumpy(X.indices)
        data_np = cp.asnumpy(X.data)
        gids_np = cp.asnumpy(gids)
        knndis_np = cp.asnumpy(knndis)
        knnidx_np = cp.asnumpy(knnidx)
        
        ''' 
        py_gpuknn(X.indptr, X.indices, X.data, gids, \
                             leaves, points_per_leaf * leaves, \
                             K, knndis.ravel(),knnidx.ravel(), \
                             maxnnz)
        '''
       
        tot_pt = points_per_leaf * leaves
        knndis_np_o, knnidx_np_o = py_FIKNN_sparse(indptr_np, indices_np, data_np, gids_np, \
                             tot_pt, leaves, \
                             K, knndis_np.ravel(), knnidx_np.ravel(), \
                             maxnnz)
        knndis_np = np.reshape(knndis_np_o, knndis_np.shape)
        knnidx_np = np.reshape(knnidx_np_o, knnidx_np.shape)
     
        toc = time()
        print('sgemm %d it too took %.2e secs'%(k, toc-tic))


if 1: # check accuracy for first leaf
    h_knndis = cp.asnumpy(knndis_np)
    h_knnidx = cp.asnumpy(knnidx_np)
    h_knndis_o = cp.asnumpy(knndis_np_o)
    h_knnidx_o = cp.asnumpy(knnidx_np_o)
     
    data    = cp.asnumpy(X.data)
    indptr  = cp.asnumpy(X.indptr)
    indices = cp.asnumpy(X.indices)
    hX = csr_matrix((data,indices,indptr), shape=(n,dim))
    # compute exact knns using sklearn
    nex = points_per_leaf
    t = 1
    nbrs = NearestNeighbors(n_neighbors=K,algorithm='brute').fit(hX[t*nex:(t+1)*nex,:])
    knndis_ex, knnidx_ex = nbrs.kneighbors(hX[t*nex:(t+1)*nex,:])
    
    print('true')
    print(knndis_ex[t, :]) 
    print(knnidx_ex[t, :]) 
    print('rec')
    #print(h_knndis[t*nex:(t+1)*nex, :])
    #print(h_knnidx[t*nex:(t+1)*nex, :])
    print(h_knndis_o.shape)
    print(h_knndis_o[t*K:(t+1)*(K)])
    print(h_knnidx_o[t*K:(t+1)*(K)])
    print('points')
    idx0_p1 = indptr[t];
    idx1_p1 = indptr[t+1];
    
    tmp = int(h_knnidx_o[t*K + 1])
    print(tmp)
    idx0_p2 = indptr[tmp]
    tmp = int(h_knnidx_o[t*K + 1]+1)
    print(tmp)
    idx1_p2 = indptr[tmp];
    
    d1 = data[idx0_p1:idx1_p1]
    d2 = data[idx0_p2:idx1_p2]
    s = np.linalg.norm(d1)**2 + np.linalg.norm(d2)**2
    print('sum norms', s**0.5)
    print('sum norms', s)
     
    print('p 1')
    print('column')
    print(indices[idx0_p1:idx1_p1])
    print('data')
    print(data[idx0_p1:idx1_p1])
     
    print('p 2')
    print('column')
    print(indices[idx0_p2:idx1_p2])
    print('data')
    print(data[idx0_p2:idx1_p2])


    #print(h_knnidx)
     
    ex = knnidx_ex
    ap = h_knnidx
    rowerr = np.any(ex[:nex,]-ap[t*nex:(t+1)*nex,],axis=1)
    rowidx = np.where(rowerr==True)
    acc = 1 - len(rowidx[0])/nex
    print('Recall accuracy:', '{:.4f}'.format(acc))




