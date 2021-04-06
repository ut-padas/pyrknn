import numpy as np
from time import time
import rkdt as rt
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import coo_matrix, csr_matrix, vstack
from sklearn.datasets import load_svmlight_file


#120 days
numdays = 1
for day in range(numdays):
    file = '/work2/00921/biros/datasets/url_svmlight/Day%d.svm'%day
#    file = 'url_svmlight/Day%d.svm'%day
    URL=load_svmlight_file(file)
    if day==0:
        X=(URL[0].tocoo()).astype(np.float32)
    else:
        tmp = (URL[0].tocoo()).astype(np.float32)
        M = max(tmp.shape[1],X.shape[1])
        X.resize(X.shape[0],M)
        tmp.resize(tmp.shape[0],M)
        X= vstack([X,tmp])
X=X.tocsr()        
LogNP = int(np.floor(np.log2(X.shape[0])))
X=X[:1<<LogNP,]



def apknnerr( ex,ap,nc):
    rowerr = np.any(ex[:nc,]-ap[:nc,],axis=1)
    rowidx = np.where(rowerr==True)
    acc = 1 - len(rowidx[0])/nc
    return acc

def apknnerr_dis(ex,ap,nc):
    err =np.linalg.norm(ex[:nc,]-ap[:nc,])/np.linalg.norm(ex[:nc,])
    return err
    
                         

def monitor(t,knnidx,knndis):
    tol = 0.9
    acc = apknnerr(knnidx_ex,knnidx,nex)
    derr =apknnerr_dis(knndis_ex,knndis,nex)
    cost = t*points_per_leaf
    print('it = ', '{:3d}'.format(t), 'Recall accuracy:', '{:.4f}'.format(acc), 'distance error = {:.4f}'.format(derr), 'cost = ',cost/n)
    break_iter = False
    break_iter =  (acc>tol or cost>n)
    return break_iter




T=20
K=16
n = X.shape[0]
dim = X.shape[1]
vltype = np.float32
X=X.astype(vltype)
LogPPL=9
depth = max(0,LogNP-LogPPL)
points_per_leaf = 1<< (LogNP-depth)
print('Number of poitns =', n, ', and the dimension =', dim)
print('Tree depth =', depth)
print('points_per_leaf =', points_per_leaf)
print('Warning depth<=dim, will use non-orthogonal directions')


nex = 10
nbrs = NearestNeighbors(n_neighbors=K,algorithm='brute').fit(X)
knndis_ex, knnidx_ex = nbrs.kneighbors(X[:nex,])


knndis = np.zeros((n,K)).astype(vltype)
knnidx = np.zeros((n,K)).astype('int32')         
gids = np.zeros(n).astype('int32')


print('Starting tree iteration')
tic = time();
rt.rkdt_a2a_it(X,gids,depth,knnidx,knndis,K,T,monitor,overlap=0)
toc = time();
print('RKDT took', '{:.2f}'.format(toc-tic), 'secs')

