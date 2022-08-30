import numpy as np
import cupy as cp
from sklearn.neighbors import NearestNeighbors
from time import time
import rkdtgpu as rt
import scipy as sp
from scipy.sparse import csr_matrix
import cupyx.scipy.sparse as cpsp
import cupy as cp

def apknnerr( ex,ap,nc):
    rowerr = cp.any(ex[:nc,]-ap[:nc,],axis=1)
    rowidx = cp.where(rowerr==True)
    acc = 1 - len(rowidx[0])/nc
    return acc

def monitor(t,knnidx,knndis):
    tol = 0.9 # set this to >1 if you want to take max iterations
    acc = apknnerr(knnidx_ex,knnidx,nex)
    cost = t*points_per_leaf
    print('it = ', '{:3d}'.format(t), 'Recall accuracy:', '{:.4f}'.format(acc), 'cost = ',cost/n)
    break_iter = False
    break_iter =  (acc>tol or cost>n)
    return break_iter



import subprocess as sp
import os

def get_gpu_memory():
  _output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]

  ACCEPTABLE_AVAILABLE_MEMORY = 1024
  COMMAND = "nvidia-smi --query-gpu=memory.free --format=csv"
  memory_free_info = _output_to_list(sp.check_output(COMMAND.split()))[1:]
  memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
  print(memory_free_values)

get_gpu_memory()





if 1: 
    np.random.seed(1)
    cp.random.seed(1)


print(cp.get_default_memory_pool().get_limit())  # 1073741824


T=200
K=32
dim = 256
avg_nnz = 100
vltype = cp.float32
LogNP =15; 
n=1<<LogNP
LogPPL=7
depth = max(0,LogNP-LogPPL)
points_per_leaf = 1<< (LogNP-depth)
print('Number of poitns =', n, ', and the dimension =', dim)
print('Tree depth =', depth)
print('points_per_leaf =', points_per_leaf)
print('Warning depth<=dim, will use non-orthogonal directions')



#X = gs.gen_random_sparse_csr_gpu(n,dim,avg_nnz)
#X = sp.sparse.random(n, dim, density=avg_nnz/dim, format='csr', dtype = np.float32)
X = cpsp.random(n, dim, density=avg_nnz/dim, format='csr', dtype = cp.float32)



data = cp.asnumpy(X.data)
indptr = cp.asnumpy(X.indptr)
indices = cp.asnumpy(X.indices)

nex = min(16,n)

hX = csr_matrix((data, indices, indptr), shape=(n,dim))
nbrs = NearestNeighbors(n_neighbors=K, algorithm='brute').fit(hX)
h_knndis_ex, h_knnidx_ex = nbrs.kneighbors(hX[:nex,])

del hX
knndis_ex = cp.asarray(h_knndis_ex)
knnidx_ex = cp.asarray(h_knnidx_ex)







knndis = 1e30*cp.ones((n,K)).astype(vltype)
knnidx = cp.ones((n,K)).astype(cp.int32)         

gids = cp.zeros(n).astype(cp.int32)

tic = time()

knnidx, knndis = rt.rkdt_a2a_it(X, gids, depth, knnidx, knndis, K, 1, monitor, 0,False)

knnidx, knndis = rt.rkdt_a2a_it(X, gids, depth, knnidx, knndis, K, T-1, monitor, 0, False)
toc = time()
print(knndis[10, :])
print(knnidx[10, :])
print(knndis_ex[10, :])
print(knnidx_ex[10, :])
e = toc - tic
print("time = %.4f "%(e));

acc = apknnerr(knnidx_ex,knnidx,nex)
print('Recall accuracy:', '{:.4f}'.format(acc), 'cost = ',T*points_per_leaf/n)

