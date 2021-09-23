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

LogNP = 25; 
n=1<<LogNP   # NUMBER of POINTS
#n = 2396160
depth = 16
leaves = 1<< depth  # POINTS per leaf


print('Generate dense array')
dim =16
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
K=4
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
a = X[1170,:]
b = X[1170:1202,]

b = np.matmul(a, b.T);
print(b[:])
#b = np.ravel(b, order ='F')
#b = b.flatten()
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





def apknnerr( ex_id,ex_dist, ap_id,ap_dist ,nc):

    '''
    rowerr = np.any(ex[:nc,]-ap[:nc,],axis=1)
    rowidx = np.where(rowerr==True)
    acc = 1 - len(rowidx[0])/nc
    '''
    err = 0.0
    for i in range(nc):
        miss_array_id = [1 if ap_id[i, j] in ex_id[i, :] else 0 for j in range(K)]
        miss_array_dist = [1 if ap_dist[i, j] <= ex_dist[i, -1] else 0 for j in range(K)]
        #err += np.sum(np.logical_or(miss_array_id, miss_array_dist))
        err += np.sum(miss_array_id)
    acc = err/(nc*K)

    return acc

def apknnerr_dis(ex,ap,nc):
    err =np.linalg.norm(ex[:nc,]-ap[:nc,])/np.linalg.norm(ex[:nc,])
    print(err)
    return err



def monitor(t,knnidx,knndis):
    tol = 0.95
    acc = apknnerr(knnidx_ex,knndis_ex, knnidx, knndis,nex)
    derr =apknnerr_dis(knndis_ex,knndis,nex)
    #derr = cp.asnumpy(derr)
    cost = t*points_per_leaf
    print('it = ', '{:3d}'.format(t), 'Recall accuracy:', '{:.4f}'.format(acc), 'distance error = {:.4f}'.format(derr), 'cost = %.4f'%cost)
    break_iter = False
    break_iter =  (acc>tol or cost>n)
    return break_iter





#for t in range(leaves): # check accuracy for first leaf
if 1:






  
  for t in range(1):
  

    nbrs = NearestNeighbors(n_neighbors=K,algorithm='brute').fit(X[t*nex:(t+1)*nex,:])
    knndis_ex, knnidx_ex = nbrs.kneighbors(X[t*nex:(t+1)*nex,:])

    apid = h_knnidx[t*nex:(t+1)*nex, :] - t * nex; 
    apdis = h_knndis[t*nex:(t+1)*nex,:]
    monitor(t, apid, apdis)

    pt = 0
    
    '''
    print(knnidx_ex[pt, :]) 
    print(knndis_ex[pt, :]) 
    print(apid[pt , :])
    print(h_knndis[t * nex + pt , :])
    
    print(apid[pt , :] - knnidx_ex[pt, :]) 
    #t = 1
    
    
    print('points')

    for i in range(nex):
      ex = knnidx_ex
      #ap = h_knnidx - t * nex 
      
      if (ex[i, :] != apid[i, :]).any():
        print(i)
        print('ex')
        print(knnidx_ex[i,:]) 
        print(knndis_ex[i,:]) 
        print('rec')
        print(apid[i,:]) 
        print(h_knndis[t * nex + i,:]) 
        print(knnidx_ex[i,:] - apid[i,:]) 
    #print(knnidx_ex[pt, :] - ap[t * nex + pt , :])
    #ap = h_knnidx
    disex = knndis_ex
    disap = h_knndis[t*nex:(t+1)*nex,:]
    err = np.linalg.norm((disex-disap).flatten())
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
