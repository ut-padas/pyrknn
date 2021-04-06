import numpy as np
import cupy as cp
from sklearn.neighbors import NearestNeighbors
from time import time
import rkdtgpu as rt


def monitor(t,knnidx,knndis):
    tol = 0.9 # set this to >1 if you want to take max iterations
    acc = apknnerr(knnidx_ex,knnidx,nex)
    cost = t*points_per_leaf
    print('it = ', '{:3d}'.format(t), 'Recall accuracy:', '{:.4f}'.format(acc), 'cost = ',cost/n)
    break_iter = False
    break_iter =  (acc>tol or cost>n)
    return break_iter


def apknnerr( ex,ap,nc):
    rowerr = cp.any(ex[:nc,]-ap[:nc,],axis=1)
    rowidx = cp.where(rowerr==True)
    acc = 1 - len(rowidx[0])/nc
    return acc


if 0: 
    np.random.seed(1)
    cp.random.seed(1)

T=4
K=4
dim = 5
vltype = cp.float32

LogNP =11; 
n=1<<LogNP
X= cp.random.randn(n,dim).astype(vltype)
LogPPL=6 #8:256
depth = max(0,LogNP-LogPPL)
points_per_leaf = 1<< (LogNP-depth)
print('Number of poitns =', n, ', and the dimension =', dim)
print('Tree depth =', depth)
print('points_per_leaf =', points_per_leaf)
print('Warning depth<=dim, will use non-orthogonal directions')


knndis = cp.zeros((n,K)).astype(vltype)
knnidx = cp.zeros((n,K)).astype(cp.int32)         
gids = cp.zeros(n).astype(cp.int32)


# --- error check
nex = min(16,n)

hX = cp.asnumpy(X)

tic =time()
nbrs = NearestNeighbors(n_neighbors=K,algorithm='brute').fit(hX)
h_knndis_ex, h_knnidx_ex = nbrs.kneighbors(hX[:nex,])
toc = time()
print('SKLEARN time {:.2f} secs',toc-tic)

knndis_ex = cp.asarray(h_knndis_ex)
knnidx_ex = cp.asarray(h_knnidx_ex)


print('Starting tree iteration')
tic = time()
rt.rkdt_a2a_it(X,gids,depth,knnidx,knndis,K,1,monitor,overlap=0)
toc = time()
print (f'First iteration without merge {toc-tic} seconds')
tic = time()
rt.rkdt_a2a_it(X,gids,depth,knnidx,knndis,K,T-1,monitor,overlap=0)
toc = time();
print (f'Search took {toc-tic} seconds')

acc = apknnerr(knnidx_ex,knnidx,nex)

print('Recall accuracy:', '{:.4f}'.format(acc), 'cost = ',T*points_per_leaf/n)

# print output
if 0: print(np.concatenate((h_knnidx.T,knnidx_ex.T)))
