import numpy as np
from time import time
import rkdt as rt
import gensparse as gs
from sklearn.neighbors import NearestNeighbors

def apknnerr( ex,ap,nc):
    rowerr = np.any(ex[:nc,]-ap[:nc,],axis=1)
    rowidx = np.where(rowerr==True)
    acc = 1 - len(rowidx[0])/nc
    return acc

def monitor(t,knnidx,knndis):
    tol = 0.9
    acc = apknnerr(knnidx_ex,knnidx,nex)
    cost = t*points_per_leaf
    print('it = ', '{:3d}'.format(t), 'Recall accuracy:', '{:.4f}'.format(acc), 'cost = ',cost/n)
    break_iter = False
    break_iter =  (acc>tol or cost>n)
    return break_iter



if 0: 
    np.random.seed(1)

T=30
K=4
dim = 300
avg_nnz = 10
vltype = np.float32
LogNP =14; 
n=1<<LogNP
LogPPL=7 
depth = max(0,LogNP-LogPPL)
points_per_leaf = 1<< (LogNP-depth)
print('Number of poitns =', n, ', and the dimension =', dim)
print('Tree depth =', depth)
print('points_per_leaf =', points_per_leaf)
print('Warning depth<=dim, will use non-orthogonal directions')



X = gs.gen_random_sparse_csr(n,dim,avg_nnz)


nex = 10
tic =time()
nbrs = NearestNeighbors(n_neighbors=K,algorithm='brute').fit(X)
knndis_ex, knnidx_ex = nbrs.kneighbors(X[:nex,])
toc = time()
print('SKLEARN time', '{:.2f}'.format(toc-tic), 'secs')


knndis = np.zeros((n,K)).astype(vltype)
knnidx = np.zeros((n,K)).astype('int32')         
gids = np.zeros(n).astype('int32')


print('Starting tree iteration')
tic = time();
rt.rkdt_a2a_it(X,gids,depth,knnidx,knndis,K,T,monitor,overlap=20)
toc = time();
print('RKDT took', '{:.2f}'.format(toc-tic), 'secs')

