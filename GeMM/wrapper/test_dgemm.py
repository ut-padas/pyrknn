import numpy as np
#import cupy as cp
from time import time
from sklearn.neighbors import NearestNeighbors
import sys

from cuda_wrapper.dense import *

mempool = cp.get_default_memory_pool()
mempool.free_all_blocks()

if 1: 
    cp.random.seed(1)

LogNP = 22; 
n=1<<LogNP   # NUMBER of POINTS

depth = 11
leaves = 1<< depth  # POINTS per leaf


print('Generate dense array')
dim = 64
X = np.random.rand(n, dim)
#np.save("tmp_mat", X)
#X = np.load("tmp_mat.npy")


n,_ = X.shape
print('Generate gids')
#gids = cp.random.permutation(cp.arange(n))
gids = np.arange(n)

points_per_leaf = int(n//leaves)
ppl = points_per_leaf
nex = points_per_leaf
'''
for i in range(leaves):
  gids[i*nex:(i+1)*nex] = np.random.permutation(gids[i*nex:(i+1)*nex])

'''

print('Generate knn arrays')
K=32
knnidx = -np.ones((n,K),dtype=np.int32)
knndis = np.ones((n,K),dtype=np.float32) + 1e30


#knndis[:,0] =0.0
#knnidx[:,0] = cp.arange(n)
leaves = 1<<depth

print('Entering SpGEMM')
print('%d total points, %d leaves, %d neighbors'%(n,leaves,K))
print('points_per_leaf =', points_per_leaf)

numits = 1

#a = np.sum(np.abs(X)**2,axis=-1)
'''
a = X[32,:]
b = X[32:ppl,]

b = np.matmul(a, b.T);
'''
#b = np.ravel(b, order ='F')
#b = b.flatten()
#print(b[:])
'''
for i in range(64):
  print("python temp_knn[%d] = %.4f"%(i, b[i]))
'''
for k in range(1):
    tic = time()
    if 1:
        knnidx_np_o , knndis_np_o = py_dfiknn(gids, X, leaves, K, knndis.ravel(), knnidx.ravel(), dim)
        
        h_knndis = np.reshape(knndis_np_o, (n,K))
        h_knnidx = np.reshape(knnidx_np_o, (n,K))
     
        toc = time()
        print('sgemm %d it too took %.2e secs'%(k, toc-tic))


#for t in range(leaves): # check accuracy for first leaf
if 1:




    t = 1



    nbrs = NearestNeighbors(n_neighbors=K,algorithm='brute').fit(X[t*nex:(t+1)*nex,:])
    knndis_ex, knnidx_ex = nbrs.kneighbors(X[t*nex:(t+1)*nex,:])






    
    #t = 1
    pt = 32
    print('true')
    
    print(knndis_ex[pt, :]) 
    print(knnidx_ex[pt, :]) 
    
   
    print('rec')
    print(h_knndis[t*nex + pt, :])
    print(h_knnidx[t * nex + pt, :] - t * nex)
    #print(h_knndis_o[t*K:(t+1)*(K)])
    #print(h_knnidx_o[t*K:(t+1)*(K)])
    print('points')

    '''
    for i in range(nex):
      if (knnidx_ex[i, :] != h_knnidx[i, :]).any():
        print(i)
        print('ex')
        print(knnidx_ex[i,:]) 
        print(knndis_ex[i,:]) 
        print('rec')
        print(h_knnidx[i,:]) 
        print(h_knndis[i,:]) 
    '''
    ex = knnidx_ex
    ap = h_knnidx - t * nex 
    #ap = h_knnidx
    disex = knndis_ex
    disap = h_knndis[t*nex:(t+1)*nex,:]
    err = np.linalg.norm((disex-disap).flatten())
    '''
    pt = 0
    print(err) 
    print('true')
    print(ex[pt,:])
    print('rec')
    print(ap[t*nex+pt,:])
    print('true')
    print(knndis_ex[pt,:])
    print('rec')
    print(h_knndis[t*nex+pt,:])
    print(ex[pt,:] - ap[t*nex+pt,:]) 
    print(knndis_ex[pt,:] - h_knndis[t*nex+pt,:])  
    '''
    rowerr = np.any(ex[:nex,]-ap[t*nex:(t+1)*nex,],axis=1)
    rowidx = np.where(rowerr==True)
    acc = 1 - len(rowidx[0])/nex
    print('Recall accuracy:', '{:.4f}'.format(acc))
    print('err:', '{:.4f}'.format(err))
