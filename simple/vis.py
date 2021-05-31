import numpy as np
import rkdt as rt
#import faiss
from sklearn.neighbors import NearestNeighbors
from time import time

def apknnerr( ex,ap,nc):
    rowerr = np.any(ex[:nc,]-ap[:nc,],axis=1)
    rowidx = np.where(rowerr==True)
    acc = 1 - len(rowidx[0])/nc
    return acc

def monitor(t,knnidx,knndis):
    acc = apknnerr(knnidx_ex,knnidx,nex)
    cost = t*points_per_leaf
    print('it = ', '{:3d}'.format(t), 'Recall accuracy:', '{:.4f}'.format(acc), 'cost = ',cost/n)
    break_iter = False
    break_iter =  (acc>tol or cost>n)
    return break_iter


#------------------------------
# 
derandomize = False
rseed = np.random.randint(100000)
if derandomize: np.random.seed(3203)

    
T = 1 #number of tree iterations
tol = 0.98
dim=2
K=128
LogNP=12
n=1<<LogNP
vltype = np.float32
X= np.random.rand(n,dim).astype(vltype)
nex = 100 # how many points to chek
LogPPL=8  #8:256
points_per_leaf = 1<< LogPPL
depth = max(0,LogNP-LogPPL)
print('Warning depth<=dim, will use non-orthogonal directions')
print('Number of poitns', n, 'dimension', dim)
print('Tree depth =', depth)
print('points_per_leaf', points_per_leaf)

tic =time()
nbrs = NearestNeighbors(n_neighbors=K,algorithm='brute').fit(X)
knndis_ex, knnidx_ex = nbrs.kneighbors(X[:nex,])
toc = time()
print('SKLEARN time', '{:.2f}'.format(toc-tic), 'secs')



if derandomize and 1:
    #np.random.seed(rseed)
    np.random.seed(38243343)


knndis = np.zeros((n,K)).astype(vltype)
knnidx = np.zeros((n,K)).astype('int32')         
gids = np.zeros(n).astype('int32')

print('Starting tree iteration')
tic = time();
rt.rkdt_a2a_it(X,gids,depth,knnidx,knndis,K,T,monitor,visualize=True)
toc = time();
print('RKDT took', '{:.2f}'.format(toc-tic), 'secs')

















                        
