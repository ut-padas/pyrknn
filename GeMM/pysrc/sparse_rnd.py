#from time import time
#import rkdt as rt
import sys
sys.path.append("dense")
sys.path.append("tree")
sys.path.append("utils")
import rkdtgpu as rt
from sklearn.neighbors import NearestNeighbors
from sklearn.datasets import load_svmlight_file

import time
import platform 
import os 
from joblib import Memory
from scipy.sparse import vstack as sparse_stack 
import argparse
from dense.dense import *
import numpy as np
import cupy as cp
import cupyx.scipy.sparse as cpsp

from utilsExact import *

parser = argparse.ArgumentParser(description="Test Sparse KNN")
parser.add_argument('-n', type=int, default=2**22)
parser.add_argument('-d', type=int, default=100)
parser.add_argument('-avgnnz', type=int, default=16)
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

def read_truth(name, k):

    id_file = name+"_nborID_100.bin.npy"
    dist_file =name+"_nborDist_100.bin.npy"

    truthID = np.load(id_file)
    truthDist = np.load(dist_file)

    print("Truth Shape: ", truthID.shape)

    truth = (truthID, truthDist)
    return truth

print("Starting Script", flush=True)
mem = Memory("./mycache")
DAT_DIR = "/scratch/07544/ghafouri/pyrknn/GeMM/datasets"


def apknnerr( ex_id,ex_dist, ap_id,ap_dist ,nc):
     
    err = 0.0
    for i in range(nc):
        miss_array_id = [1 if ap_id[i, j] in ex_id[i, :] else 0 for j in range(K)]
        miss_array_dist = [1 if ap_dist[i, j] <= ex_dist[i, -1] else 0 for j in range(K)]
        print(i)
        #err += np.sum(np.logical_or(miss_array_id, miss_array_dist))
        err += cp.sum(cp.asarray(miss_array_id))
    acc = err/(nc*K)

    return acc

def apknnerr_dis(ex,ap,nc):
    err =cp.linalg.norm(ex[:nc,]-ap[:nc,])/cp.linalg.norm(ex[:nc,])
    return err
    
                         

def monitor(t,knnidx,knndis):
    tol = 0.95
    knnidx = cp.array(knnidx)
    knndis = cp.array(knndis)
    acc = apknnerr(knnidx_ex,knndis_ex, knnidx, knndis,nex)
    derr = apknnerr_dis(knndis_ex,knndis,nex)
    derr = cp.asnumpy(derr)
    cost = t*points_per_leaf
    print('it = ', '{:3d}'.format(t), 'Recall accuracy:', '{:.4f}'.format(acc), 'distance error = {:.4f}'.format(derr), 'cost = %.4f'%cost)
    break_iter = False
    break_iter =  (acc>tol or cost>n)
    return break_iter



dataset = args.dataset
name = dataset
n = args.n
dim = args.d
avgnnz = args.avgnnz
K = args.k
T = args.iter

X = cpsp.random.randn(n,dim, density=avgnnz/dim, format='csr', dtype = cp.float32)

knndis = 1e30*cp.ones((n,K), dtype = cp.float32)
knnidx = -cp.ones((n,K), dtype = cp.int32)         


print("Finished Reading Data", flush=True)
depth = args.levels

#X = X[:n,]

N  = X.shape[0]
d  = X.shape[1]
print("Init Data shape: ", (N, d))

cp.random.seed(args.seed)

print('Padding the data')

leaves = 1 << depth 
ppl = cp.ceil(n / leaves)
n_true = int(ppl * leaves)
diff = n_true - n
if diff > 0:
  X = cp.pad(X, (0, diff), "constant")
  n, dim = X.shape

points_per_leaf = int(n/leaves)


print('Number of poitns =', n, ', and the dimension =', dim)
print('Tree depth =', depth)
print('points_per_leaf =', points_per_leaf)
print('Warning depth<=dim, will use non-orthogonal directions')

nex = points_per_leaf
t = 0
#nbrs = NearestNeighbors(n_neighbors=K,algorithm='brute').fit(X)
'''
ex_dir = name + "ex_dense"

if not os.path.isdir(ex_dir):
  os.mkdir(ex_dir)

fname = ex_dir + "/" + name + "_knnidx_ex_L%d.npy"%depth
print(fname)
'''
print("computing the exact neghobors")
#nbrs = NearestNeighbors(n_neighbors=K,algorithm='brute').fit(cp.asnumpy(X))
#knndis_ex, knnidx_ex = nbrs.kneighbors(cp.asnumpy(X[:nex,]))
#knnidx_ex , knndis_ex = neighbors(X, leaves, K)


print('Starting tree iteration')



tic = time.time();
leaves = int(n // points_per_leaf)

#knnidx, knndis = py_dfiknn(gids, X, leaves, K, knnidx, knndis, dim)
#knnidx, knndis = py_dfiknn(gids, X, leaves, K, knnidx.ravel(), knndis.ravel(), dim)

#knnidx, knndis = py_dfiknn(gids, X, leaves, K, knnidx.ravel(), knndis.ravel(), dim)
knnidx, knndis = rt.rkdt_a2a_it(X,depth,knnidx, knndis, K,T,None,0, True)
#knnidx, knndis = rt.rkdt_a2a_it(X,gids,depth,knnidx, knndis, K,T,monitor,1, True)
toc = time.time();
print('RKDT took', '{:.2f}'.format(toc-tic), 'secs \n')
monitor(0,knnidx,knndis)
