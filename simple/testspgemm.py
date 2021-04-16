import numpy as np
import cupy as cp
from time import time
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

import sys
sys.path.append("../SpGeMM")
import SpGeMM_2D_NoPerm as sgemm


if 1: 
    cp.random.seed(1)

LogNP =21; 
n=1<<LogNP   # NUMBER of POINTS

LogPPL=11
depth = max(0,LogNP-LogPPL)
points_per_leaf = 1<< (LogNP-depth)  # POINTS per leaf


print('Generate sparse array')
dim = 100
avg_nnz = 100
X = cp.sparse.random(points_per_leaf,dim, density = avg_nnz/dim, format='csr', dtype=cp.float32)

X.sort_indices()

print('Generate gids')
gids = cp.random.permutation(cp.arange(n))

print('Generate knn arrays')
K=32
knnidx = cp.random.randint(n, size=(n,K),dtype=cp.int32)
print(knnidx.shape)
knndis = 1e30*cp.ones((n,K),dtype=cp.float32)
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
        sgemm.gpu_sparse_knn(X.indptr, X.indices, X.data,\
                             leaves, points_per_leaf, \
                             knndis.ravel(),knnidx.ravel(), K, 
                             8,1,2,1024, 2e30, maxnnz)

        toc = time()
        print('sgemm %d it too took %.2e secs'%(k, toc-tic))


if 1: # check accuracy for first leaf
    h_knndis = cp.asnumpy(knndis)
    h_knnidx = cp.asnumpy(knnidx)
     
    data    = cp.asnumpy(X.data)
    indptr  = cp.asnumpy(X.indptr)
    indices = cp.asnumpy(X.indices)
    hX = csr_matrix((data,indices,indptr), shape=(n,dim))
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



