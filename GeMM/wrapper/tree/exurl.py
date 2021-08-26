import numpy as np
#from time import time
#import rkdt as rt
import rkdtgpu as rt
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import coo_matrix, csr_matrix, vstack
from sklearn.datasets import load_svmlight_file

#from mpi4py import MPI
import numpy as np

import time
import platform 
import os 
from joblib import Memory
from scipy.sparse import vstack as sparse_stack 
import argparse
from cuda_wrapper.sparse import *
import cupy as cp
import sys
sys.path.append("../../wrapper")

parser = argparse.ArgumentParser(description="Test Sparse KNN")
parser.add_argument('-n', type=int, default=2**22)
parser.add_argument('-d', type=int, default=15)
parser.add_argument('-iter', type=int, default=120)
parser.add_argument('-dataset', default="gauss")
parser.add_argument('-bs', type=int, default=64)
parser.add_argument('-bl', type=int, default=128)
parser.add_argument('-cores', type=int, default=56)
parser.add_argument('-use_gpu', type=bool, default=0)
parser.add_argument('-levels', type=int, default=13)
parser.add_argument('-k', type=int, default=32)
parser.add_argument('-leafsize', type=int, default=1024)
parser.add_argument('-ltrees', type=int, default=1)
parser.add_argument('-q', type=int, default=100)
parser.add_argument('-merge', type=int, default=1)
parser.add_argument('-overlap', type=int, default=0)
parser.add_argument('-seed', type=int, default=15)
parser.add_argument('-nq', type=int, default=1000)
args = parser.parse_args()

if args.use_gpu:
    location = "GPU"
else:
    location = "HOST"
'''
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
'''
size = 1
rank = 0
K = 64
t = 0
nq = 100


def read_truth(name, k):

    id_file = name+"_nborID_100.bin.npy"
    dist_file =name+"_nborDist_100.bin.npy"

    #truthID = np.fromfile(id_file, dtype=np.int32)
    #truthDist = np.fromfile(dist_file, dtype=np.float32)
    truthID = np.load(id_file)
    truthDist = np.load(dist_file)

    #truthID = truthID.reshape((len(truthID)//k, k))
    #truthDist = truthID.reshape(truthID.shape)
    print("Truth Shape: ", truthID.shape)

    truth = (truthID, truthDist)
    return truth

print("Starting Script", flush=True)
mem = Memory("./mycache")
name = os.environ["SCRATCH"]+"/comparison/avazu/"
name = "avazu"

@mem.cache()
def get_data():
    t = time.time()
    prefix = "/work2/07544/ghafouri/frontera/gits/pyrknn/"
    data_app = load_svmlight_file(prefix + "/datasets/avazu/avazu-app", n_features=1000000, dtype=np.float32)
    data_site = load_svmlight_file(prefix+"/datasets/avazu/avazu-site", n_features=1000000, dtype=np.float32)
    print(data_app[0], data_app[1])
    print(data_app[0].shape, data_app[1].shape)
    t = time.time() - t
    print("It took ", t, " (s) to load the dataset")
    return data_app[0], data_site[0]

print("Starting to Read Data", flush=True)
X_app, X_site = get_data()
print(X_app.shape, X_site.shape)
X = sparse_stack([X_app, X_site])


print("Finished Reading Data", flush=True)
k = 64
n = args.n
X = X[:n,]
N  = X.shape[0]
d  = X.shape[1]

print("Data shape: ", (N, d))
#truth = read_truth(name, k)

local_size = N//size
X = X[rank*local_size:(rank+1)*local_size]

#timer = Profiler()
#record = Recorder()

np.random.seed(args.seed)



'''


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
'''


def apknnerr( ex,ap,nc):
    
    rowerr = cp.any(ex[:nc,:] - ap[:nc,:],axis=1)
    rowidx = cp.where(rowerr==True)
    acc = 1 - len(rowidx[0])/nc
    return acc

def apknnerr_dis(ex,ap,nc):
    err =np.linalg.norm(ex[:nc,]-ap[:nc,])/np.linalg.norm(ex[:nc,])
    return err
    
                         

def monitor(t,knnidx,knndis):
    tol = 0.95
    acc = apknnerr(knnidx_ex,knnidx,nex)
    derr =apknnerr_dis(knndis_ex,knndis,nex)
    derr = cp.asnumpy(derr)
    cost = t*points_per_leaf
    print('it = ', '{:3d}'.format(t), 'Recall accuracy:', '{:.4f}'.format(acc), 'distance error = {:.4f}'.format(derr), 'cost = %.4f'%cost)
    break_iter = False
    break_iter =  (acc>tol or cost>n)
    return break_iter



'''
T=20
K=16
n = X.shape[0]
dim = X.shape[1]
vltype = np.float32
X=X.astype(vltype)
LogPPL=9
depth = max(0,LogNP-LogPPL)
points_per_leaf = 1<< (LogNP-depth)
'''

depth = args.levels
points_per_leaf = args.leafsize
T = args.iter
dim = X.shape[1]
n = X.shape[0]

print('Number of poitns =', n, ', and the dimension =', dim)
print('Tree depth =', depth)
print('points_per_leaf =', points_per_leaf)
print('Warning depth<=dim, will use non-orthogonal directions')



nex = points_per_leaf
t = 10
nbrs = NearestNeighbors(n_neighbors=K,algorithm='brute').fit(X[t*nex:(t+1)*nex, ])
knndis_ex, knnidx_ex = nbrs.kneighbors(X[t*nex:(t+1)*nex,])
knndis_ex = cp.asarray(knndis_ex)
knnidx_ex = cp.asarray(knnidx_ex)

knndis = cp.ones((n,K), dtype = np.float32) + 1e38
knnidx = cp.zeros((n,K), dtype = np.int32)         
#gids = np.zeros(n).astype('int32')
gids = cp.zeros(n).astype(cp.int32)

print('Starting tree iteration')
data = X.data
print(type(data[0]))
tic = time.time();
knnidx, knndis = rt.rkdt_a2a_it(X,gids,depth,knnidx,knndis,K,T,monitor,0, False)
toc = time.time();
print('RKDT took', '{:.2f}'.format(toc-tic), 'secs')
'''
if rank == 0:
  timer.print()
  record.print()

'''
