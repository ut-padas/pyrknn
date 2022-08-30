#from time import time
#import rkdt as rt
import sys
import filknn.tree.rkdtgpu as rt
from sklearn.neighbors import NearestNeighbors
from sklearn.datasets import load_svmlight_file

import time
import platform 
import os 
from joblib import Memory
import argparse
from filknn.sparse.sparse import *
import numpy as np
import cupy as cp
import scipy.sparse as sp
import cupy.random 
import cupyx.scipy.sparse as cpsp
from filknn.utils.utilsExact import *
from datetime import datetime
from scipy.sparse import csr_matrix

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


print("Starting Script", flush=True)
mem = Memory("./mycache")


def apknnerr( ex_id,ex_dist, ap_id,ap_dist ,nc):

    err = 0.0
    #for i in range(nc):
    for (i,ptid) in enumerate(test_pt):
      #miss_array_dist = cp.zeros(K)
      miss_array_id = cp.zeros(K)
      for j in range(K):
        #if ap_dist[ptid,j] <= ex_dist[i, -1]:
        #  miss_array_dist[j] = 1
        #  #  print(nc, K)
        if ap_id[ptid,j] in ex_id[i, :]:
          miss_array_id[j] = 1
        '''
        else:
          #print(ptid, j, ap_dist[ptid, j], ex_dist[i, -1])
          print(ptid, j, ap_id[ptid, :], ex_id[i, :])
          break
        '''
        #print(i)
        #miss_array_id = cp.asarray(miss_array_id)
        #miss_array_dist = cp.asarray(miss_array_dist)
        #err += cp.sum(cp.logical_or(miss_array_id, miss_array_dist))
      err += cp.sum(cp.asarray(miss_array_id))
      #err += cp.sum(miss_array_dist)
    acc = err/(nc*K)

    return acc

def apknnerr_dis(ex,ap,nc):
    err = cp.linalg.norm(ex[:nc,]-ap[test_pt,])/cp.linalg.norm(ex[test_pt,])
    return err



def monitor(t,knnidx,knndis):
    tol = 0.95
    num_test = test_pt.shape[0]
    #knnidx = cp.array(knnidx)
    #knndis = cp.array(knndis)
    acc = apknnerr(knnidx_ex,knndis_ex, knnidx, knndis,num_test)
    derr = apknnerr_dis(knndis_ex,knndis,num_test)
    #derr = cp.asnumpy(derr)
    cost = t*ppl
    print('it = ', '{:3d}'.format(t), 'Recall accuracy:', '{:.4f}'.format(acc), 'distance error = {:.4f}'.format(derr), 'cost = %.4f'%cost)
    break_iter = False
    break_iter =  (acc>tol or cost>n)
    return break_iter

@mem.cache()
def get_url_data():
    t = time.time()
    data = load_svmlight_file(DAT_DIR+"/url/url_combined", n_features=3231961)
    t = time.time() - t
    print("It took ", t, " (s) to load the dataset")
    return data[0]


@mem.cache()
def get_avazu_data():
    t = time.time()
    data_app = load_svmlight_file(DAT_DIR+"/avazu/avazu-app", n_features=1000000)
    print(data_app[0].shape)
    data_site = load_svmlight_file(DAT_DIR+"/avazu/avazu-site", n_features=1000000)
    print(data_site[0].shape)
    data = sp.vstack([data_app[0], data_site[0]])
    t = time.time() - t
    print("It took ", t, " (s) to load the dataset")
    data = cpsp.csr_matrix((cp.array(data.data), (data.indices, data.indptr)), data.shape, format='csr')
    return data




dataset = args.dataset
name = dataset
n = args.n
dim = args.d
avgnnz = args.avgnnz
K = args.k
T = args.iter
depth = args.levels

DAT_DIR='/scratch/07544/ghafouri/pyrknn/GeMM/datasets'

if dataset == 'url':
  X = get_url_data()
  
  v = cp.array(X.data, dtype = cp.float32)
  idx = cp.array(X.indices, dtype = cp.int32)
  rowptr = cp.array(X.indptr, dtype = cp.int32)
  X = cpsp.csr_matrix((v, idx, rowptr))
  del v
  del idx
  del rowptr
  n,dim = X.shape
elif dataset == 'avazu':
  X = get_avazu_data()
  n,dim = X.shape
else:
  X = cpsp.random(n,dim, density=avgnnz/dim, format='csr', dtype = cp.float32)


knndis = 1e30*cp.ones((n,K), dtype = cp.float32)
knnidx = -cp.ones((n,K), dtype = cp.int32)         


print("Finished Reading Data", flush=True)

#X = X[:n,]

N  = X.shape[0]
d  = X.shape[1]
print("Init Data shape: ", (N, d))

cp.random.seed(args.seed)

print('Padding the data')

leaves = 1 << depth 
ppl = cp.ceil(n / leaves)
n_true = cp.array(ppl * leaves)
diff = n_true - N
last = cp.asarray(X.indptr[N])
avgnnz = cp.mean(cp.diff(X.indptr))
print("avgnnz = %d "%avgnnz)
print(X.data.shape)
print(X.indices.shape)
print(X.indptr.shape)
pad_width = np.zeros(2, dtype=cp.int32)
end_val = np.zeros(2, dtype=cp.int32)
pad_width[1] = diff
end_val[1] = last


if diff > 0:
  #tmp = cp.pad(X.indptr, pad_width, "linear_ramp", end_values=end_val)
  X.indptr = cp.pad(X.indptr, pad_width, "edge")
   
  X = cpsp.csr_matrix((X.data, X.indices, X.indptr))
  #X = cp.pad(X, (0,diff), "constant")
  n, dim = X.shape
  

ppl = int(n/leaves)


print('Number of poitns =', n, ', and the dimension =', dim)
print('Tree depth =', depth)
print('pts/leaf =', ppl)
print('Warning depth<=dim, will use non-orthogonal directions')

#nbrs = NearestNeighbors(n_neighbors=K,algorithm='brute').fit(X)
print('Starting tree iteration')

tic = time.time();
'''
gids = cp.arange(0, n, dtype = cp.int32)
for i in range(leaves):
  gids[i*ppl:(i+1)*ppl] = cp.random.permutation(gids[i*ppl:(i+1)*ppl])

knnidx, knndis = py_sfiknn(gids, X, leaves, K, knndis, knnidx)
'''
knnidx, knndis = rt.rkdt_a2a_it(X,depth,knnidx, knndis, K,T,None,0, False)
toc = time.time();



print('RKDT took', '{:.2f}'.format(toc-tic), 'secs \n')

#cp.save("results/sparse/knndis", knndis)
#cp.save("results/sparse/knnidx", knnidx)

#knndis = cp.load("../results/sparse/knndis.npy")
#knnidx = cp.load("../results/sparse/knnidx.npy")



print("computing the exact neghobors")
'''
for t in range(331, 332, leaves):
  hX = csr_matrix((cp.asnumpy(X.data), cp.asnumpy(X.indices), cp.asnumpy(X.indptr)), shape=(n,dim))
  nbrs = NearestNeighbors(n_neighbors=K,algorithm='brute').fit(hX[t*ppl:(t+1)*ppl,])
  knndis_ex, knnidx_ex = nbrs.kneighbors(hX[t*ppl:(t+1)*ppl,:])
  knndis_ex = cp.asarray(knndis_ex)
  knnidx_ex = cp.asarray(knnidx_ex)
  test_pt = cp.arange(t*ppl, (t+1)*ppl)
  monitor(t,knnidx - t * ppl,knndis)

'''
#test_pt = cp.arange(0, 1000)
fname = '../results/' + dataset + '_ex/'

knnidx_ex = cp.array(cp.load(fname + 'knnId_ex.npy'), dtype = cp.int32)
knndis_ex = cp.array(cp.load(fname + 'knnDist_ex.npy'), dtype = cp.float32)**0.5
test_pt = cp.array(cp.load(fname + 'test_pt.npy'), dtype = cp.int32)
knndis_ex = cp.nan_to_num(knndis_ex)
monitor(0, knnidx, knndis)

'''
for i in range(1):
  cp.random.seed(i)
  test_pt = cp.random.randint(0, N, 32)
  knnidx_ex , knndis_ex = neighbors(X, K, test_pt)
  cp.save(fname + "/Dist_ex_%d"%i, knndis_ex)
  cp.save(fname + "/Id_ex_%d"%i, knnidx_ex)
  cp.save(fname + "/test_pt_%d"%i, test_pt)
  del knnidx_ex
  del knndis_ex
  del test_pt
test_pt = cp.arange(t*ppl, (t+1)*ppl)

pt = 0
l = 331 * ppl
test_pt = cp.arange(l, l + ppl)
monitor(0, knnidx - l, knndis)
print(knnidx[l + pt, :]- l)
print(knnidx_ex[pt, :] )
print(knndis[l + pt, :])
print(knndis_ex[pt, :])

'''
