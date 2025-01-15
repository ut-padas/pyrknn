#from time import time
#import rkdt as rt
import sys
import filknn.tree.rkdtgpu as rt
from sklearn.neighbors import NearestNeighbors
from sklearn.datasets import load_svmlight_file
import math 
import time
import platform 
import os 
from joblib import Memory
from scipy.sparse import vstack as sparse_stack 
import argparse
from filknn.dense.dense import *
import numpy as np
import cupy as cp
from filknn.utils.utilsExact import *

parser = argparse.ArgumentParser(description="Test Sparse KNN")
parser.add_argument('-n', type=int, default=1000000)
parser.add_argument('-d', type=int, default=15)
parser.add_argument('-iter', type=int, default=120)
parser.add_argument('-dataset', default="gauss")
parser.add_argument('-bs', type=int, default=64)
parser.add_argument('-bl', type=int, default=128)
parser.add_argument('-cores', type=int, default=56)
parser.add_argument('-use_gpu', type=bool, default=0)
parser.add_argument('-levels', type=int, default=9)
parser.add_argument('-k', type=int, default=32)
parser.add_argument('-leafsize', type=int, default=1024)
parser.add_argument('-ltrees', type=int, default=1)
parser.add_argument('-q', type=int, default=100)
parser.add_argument('-merge', type=int, default=1)
parser.add_argument('-overlap', type=int, default=0)
parser.add_argument('-seed', type=int, default=15)
parser.add_argument('-nq', type=int, default=1000)
args = parser.parse_args()


def read_sift(d):

  filename='dataset/sift/sift_learn.fvecs'
  vsz = 4 + d
  nc = 2
  v = cp.fromfile(filename, dtype=cp.uint8, count=nc*vsz, offset=st*vsz)
  X = v.reshape((nc, d+4))
  
  return X  

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
K = args.k
T = args.iter
depth = args.levels
leafsize = args.leafsize
nq = args.nq
if dataset == 'sift':
  X = read_sift(d)
elif dataset == 'gaussian':
  X = read_gaussian(n,dim)
else:
  X = read_uniform(n,dim)


knndis = 1e30*cp.ones((n,K), dtype = cp.float32)
knnidx = -cp.ones((n,K), dtype = cp.int32)         


print("Finished Reading Data", flush=True)

n  = X.shape[0]
d  = X.shape[1]

print("Init Data shape: ", (n, d))

cp.random.seed(args.seed)

print('Padding the data')



def find_p_k_L(Z, N, max_L=64):
    if N == 0 and Z == 0:
        return 0, 0, 0
    best, best_dist = (float("inf"), 0, 0), float("inf")
    for L in range(max_L + 1):
        # Minimal M to satisfy 2^L * M >= N
        M_needed = 0 if N == 0 else math.ceil(N / (1 << L))
        # Check multiples of 32 around M_needed (floor & ceil)
        floor_32 = 32 * (M_needed // 32)
        ceil_32 = 32 * ((M_needed + 31) // 32)
        for M in (floor_32, ceil_32):
            if M < 0:
                continue
            if Z > 0 and not (Z / 2 <= M <= 2 * Z):
                continue
            p = (1 << L) * M - N
            if p < 0:
                continue
            dist = abs(M - Z)
            if p < best[0] or (p == best[0] and dist < best_dist):
                best, best_dist = (p, M // 32, L), dist
    return best
  
p_best, k_best, L_best = find_p_k_L(leafsize, n, max_L=32)


ppl = k_best * 32
depth = L_best


if p_best > 0:
  print("PADDING", p_best)
  padding = np.zeros((p_best, d), dtype=np.float32)
  X = cp.vstack([X, padding])
  n, dim = X.shape
  print("X shape after padding: ", X.shape)
points_per_leaf = ppl 

print('Number of poitns =', n, ', and the dimension =', dim)
print('Tree depth =', depth)
print('points_per_leaf =', points_per_leaf)
print('Warning depth<=dim, will use non-orthogonal directions')

nex = points_per_leaf

print("computing the exact neghobors")
#nbrs = NearestNeighbors(n_neighbors=K,algorithm='brute').fit(cp.asnumpy(X))
#knndis_ex, knnidx_ex = nbrs.kneighbors(cp.asnumpy(X[:nex,]))

test_pt = cp.random.randint(0, n, size=nq)
knnidx_ex , knndis_ex = neighbors(X, K, test_pt)


print('Starting tree iteration')


tic = time.time();
leaves = int(n // points_per_leaf)

#knnidx, knndis = py_dfiknn(gids, X, leaves, K, knnidx, knndis, dim)
knnidx, knndis = rt.rkdt_a2a_it(X,depth,knnidx, knndis, K,T,None,0, True)

toc = time.time();
print('RKDT took', '{:.2f}'.format(toc-tic), 'secs \n')
tic = time.time()

monitor(0,knnidx,knndis)
toc = time.time() - tic

print("monitor takes %.4f \n\n"%toc)

print(knnidx_ex[0, :])
print(knnidx[test_pt[0], :])
print(knndis_ex[0, :])
print(knndis[test_pt[0], :])

