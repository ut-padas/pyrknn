import cupy as cp
from time import time

import sys
sys.path.append("../SpGeMM")
import SpGeMM_2D as sgemm


if 1: 
    cp.random.seed(1)



LogNP =21; 
n=1<<LogNP   # NUMBER of POINTS

LogPPL=11
depth = max(0,LogNP-LogPPL)
points_per_leaf = 1<< (LogNP-depth)  # POINTS per leaf


print('Generate sparse array')
dim = 100
avg_nnz = 20
X = cp.sparse.random(n,dim, density = avg_nnz/dim, format='csr', dtype=cp.float32)
X.sort_indices()

print('Generate gids')
gids = cp.random.permutation(cp.arange(n))

print('Generate knn arrays')
K=32
knnidx = cp.random.randint(n, size=(n,K),dtype=cp.int32)
knndis = 1e30*cp.ones((n,K),dtype=cp.float32)
knndis[:,0] =0.0
knnidx[:,0] = cp.arange(n)
maxnnz = cp.max(cp.diff(X.indptr))
leaves = 1<<depth

print('Entering SpGEMM')
print('%d total points, %d leaves, %d neighbors'%(n,leaves,K))
print('points_per_leaf =', points_per_leaf)
for k in range(2):
    tic = time()
    if 1:
        sgemm.gpu_sparse_knn(X.indptr, X.indices, X.data, gids,\
                             leaves, n, \
                             knndis.ravel(),knnidx.ravel(), K, 
                             32,32,64,64, 2e30, maxnnz)

        toc = time()
        print('sgemm %d it too took %.2e secs'%(k, toc-tic))

