#from time import time
#import rkdt as rt
import sys
sys.path.append("../dense")
sys.path.append("../tree")
sys.path.append("../utils")
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
from utilsExact import *

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


def read_sift():

  filename='../dataset/sift/sift_base.fvecs'
  c_contiguous = True
  fv = cp.fromfile(filename, dtype=cp.float32)
  print(fv.shape)
  if fv.size == 0:
    return np.zeros((0, 0))
  dim = fv.view(cp.int32)[0]
  assert dim > 0
  n = int(fv.shape[0] // (1+dim))
  fv = cp.reshape(fv, (n,int(1+dim)))
  #fv = fv.reshape((n, 1 + dim))
  if not all(fv.view(np.int32)[:, 0] == dim):
    raise IOError("Non-uniform vector sizes in " + filename)
  fv = fv[:, 1:]
  if c_contiguous:
    fv = fv.copy()
  return fv


def read_gaussian(n, dim):

  X = cp.random.randn(n,dim, dtype = cp.float32)
  return X

def read_uniform(n, dim):
  
  X = cp.random.rand(n,dim, dtype = cp.float32)
  
  return X

print("Starting Script", flush=True)
mem = Memory("./mycache")

def apknnerr( ex_id,ex_dist, ap_id,ap_dist ,nc):
     
    err = 0.0
    #for i in range(nc):
    for (i,ptid) in enumerate(test_pt):
      miss_array_dist = cp.zeros(K)
      #miss_array_id = cp.zeros(K)
      for j in range(K):
        if ap_dist[ptid,j] <= ex_dist[i, -1]:
          miss_array_dist[j] = 1
        #if ap_id[ptid,j] in ex_id[i, :]:
        #  miss_array_id[j] = 1
        #print(i)
        #miss_array_id = cp.asarray(miss_array_id)
        #miss_array_dist = cp.asarray(miss_array_dist)
        #err += cp.sum(cp.logical_or(miss_array_id, miss_array_dist))
      #err += cp.sum(cp.asarray(miss_array_id))
      err += cp.sum(miss_array_dist)
    acc = err/(nc*K)

    return acc

def apknnerr_dis(ex,ap,nc):
    err =cp.linalg.norm(ex[:nc,]-ap[test_pt,])/cp.linalg.norm(ex[test_pt,])
    return err
    
                         

def monitor(t,knnidx,knndis):
    tol = 0.95
    num_test = test_pt.shape[0]
    knnidx = cp.array(knnidx)
    knndis = cp.array(knndis)
    acc = apknnerr(knnidx_ex,knndis_ex, knnidx, knndis,num_test)
    derr = apknnerr_dis(knndis_ex,knndis,num_test)
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
d = args.d
K = args.k
T = args.iter
depth = args.levels
nq = args.nq
if dataset == 'sift':
  X = read_sift()
elif dataset == 'gaussian':
  X = read_gaussian(n,dim)
else:
  X = read_uniform(n,dim)



knndis = 1e30*cp.ones((n,K), dtype = cp.float32)
knnidx = -cp.ones((n,K), dtype = cp.int32)         


print("Finished Reading Data", flush=True)

N  = X.shape[0]
d  = X.shape[1]
n = N
print("Init Data shape: ", (N, d))

cp.random.seed(args.seed)

print('Padding the data')

leaves = 1 << depth 
ppl = cp.ceil(n / leaves)
n_true = int(ppl * leaves)
diff = n_true - n
print(diff, n_true, n, ppl)
if diff > 0:
  X = cp.pad(X, ((0, diff), (0,0)), "constant")
  n, dim = X.shape

points_per_leaf = int(n/leaves)

print('Number of points =', n, ', and the dimension =', dim)
print('Tree depth =', depth)
print('points_per_leaf =', points_per_leaf)
print('Warning depth<=dim, will use non-orthogonal directions')

nex = points_per_leaf

print("computing the exact neghobors")
#nbrs = NearestNeighbors(n_neighbors=K,algorithm='brute').fit(cp.asnumpy(X))
#knndis_ex, knnidx_ex = nbrs.kneighbors(cp.asnumpy(X[:nex,]))



print('Starting tree iteration')


tic = time.time();
leaves = int(n // points_per_leaf)
gids = cp.arange(0, N, dtype = cp.int32)


knnidx, knndis = py_dfiknn(gids, X, leaves, K, knnidx, knndis, dim)
#knnidx, knndis = rt.rkdt_a2a_it(X,depth,knnidx, knndis, K,T,None,0, True)

toc = time.time();
print('RKDT took', '{:.2f}'.format(toc-tic), 'secs \n')
tic = time.time()
'''
#test_pt = cp.random.randint(0, N, size=nq)
#test_pt = cp.random.randint(0, ppl, size=ppl)
test_pt = cp.arange(0, ppl)
knnidx_ex , knndis_ex = neighbors(X[:ppl, :], K, test_pt)

monitor(0,knnidx,knndis)
toc = time.time() - tic

print("monitor takes %.4f \n\n"%toc)
pt = 1
print(knnidx_ex[1, :])
print(knnidx[test_pt[1], :])
print(knnidx[test_pt[1], :] - knnidx_ex[1, :])
print(knndis_ex[1, :])
print(knndis[test_pt[1], :])
'''
