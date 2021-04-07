import numpy as np
import cupy as cp
from sklearn.neighbors import NearestNeighbors
from time import time
import rkdtgpu as rt
import gensparse as gs


if 1: 
    np.random.seed(1)
    cp.random.seed(1)

T=4
K=2
dim = 300
avg_nnz = 100
vltype = cp.float32
LogNP =20; 
n=1<<LogNP
LogPPL=10 
depth = max(0,LogNP-LogPPL)
points_per_leaf = 1<< (LogNP-depth)
print('Number of poitns =', n, ', and the dimension =', dim)
print('Tree depth =', depth)
print('points_per_leaf =', points_per_leaf)
print('Warning depth<=dim, will use non-orthogonal directions')



X = gs.gen_random_sparse_csr_gpu(n,dim,avg_nnz)
knndis = cp.zeros((n,K)).astype(vltype)
knnidx = cp.zeros((n,K)).astype(cp.int32)         
gids = cp.zeros(n).astype(cp.int32)


rt.rkdt_a2a_it(X,gids,depth,knnidx,knndis,K,1,dense=False)
