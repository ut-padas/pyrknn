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

if 1: np.random.seed(1)

    
T = 4 #number of tree iterations
tol = 0.98
dim=3
K=4
LogNP=13
n=1<<LogNP
vltype = np.float32
X= np.random.randn(n,dim).astype(vltype)
nex = 100 # how many points to chek
LogPPL=8   #8:256
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


# tic=time()
# faiss_index = faiss.IndexFlatL2(dim)
# faiss_index.add(np.float32(X))
# #
#dknn,iknn =  faiss_index.search(np.float32(X[:nex,:]),K)
# dknn,iknn =  faiss_index.search(np.float32(X),K)
# toc=time()
# print('faiss accuracy', apknnerr(knnidx_ex,iknn,nex))
# print('FAISS time', '{:.2f}'.format(toc-tic), 'secs')


knndis = np.zeros((n,K)).astype(vltype)
knnidx = np.zeros((n,K)).astype('int32')         
gids = np.zeros(n).astype('int32')

print('Starting tree iteration')
tic = time();
rt.rkdt_a2a_it(X,gids,depth,knnidx,knndis,K,T,monitor)
toc = time();
print('RKDT took', '{:.2f}'.format(toc-tic), 'secs')

















                        
