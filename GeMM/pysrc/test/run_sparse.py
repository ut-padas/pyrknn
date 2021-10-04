#from time import time
#import rkdt as rt
import sys
sys.path.append("../sparse")
sys.path.append("../tree")
sys.path.append("../utils")
import rkdtgpu as rt
from sklearn.neighbors import NearestNeighbors
from sklearn.datasets import load_svmlight_file

import time
import platform 
import os 
from joblib import Memory
import argparse
from sparse.sparse import *
import numpy as np
import cupy as cp
import scipy.sparse as sp
import cupy.random 
import cupyx.scipy.sparse as cpsp
from utilsExact import *
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
    for i in range(nc):
      miss_array_id = [1 if ap_id[test_pt[i],j] in ex_id[i,:] else 0 for j in range(K)]
      miss_array_dist = [1 if ap_dist[test_pt[i],j] <= ex_dist[i,-1]+1e-7 else 0 for j in range(K)]
      err += np.sum(np.logical_or(miss_array_id, miss_array_dist))
    hit_rate = err/(nc*K)
    mean_sim = np.mean(ap_dist.ravel())
    #last_array = np.abs(ap_dist[test_pt,-1] - ex_dist[:,-1])/ex_dist[:,-1]
    #mean_rel_err = np.mean(last_array)
    return hit_rate, mean_sim

def apknnerr_dis(ex,ap,nc):
    err = np.linalg.norm(ex[:nc,]-ap[test_pt,])/np.linalg.norm(ex[:nc,])
    return err



def monitor(t,knnidx,knndis):
    tol = 0.95
    num_test = test_pt.shape[0]
    knnidx = cp.asnumpy(knnidx)
    knndis = cp.asnumpy(knndis)
    acc, _= apknnerr(knnidx_ex,knndis_ex, knnidx, knndis,num_test)
    derr = apknnerr_dis(knndis_ex,knndis,num_test)
    #derr = cp.asnumpy(derr)
    cost = t*ppl
    print('it = ', '{:3d}'.format(t), 'Recall accuracy:', '{:.4f}'.format(acc), 'mean rel distance error = {:.4f}'.format(derr), 'cost = %.4f'%cost)
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
    datah = sp.vstack([data_app[0], data_site[0]])
    t = time.time() - t
    print("It took ", t, " (s) to load the dataset")
    v = cp.array(datah.data, dtype = cp.float32)
    idx = cp.array(datah.indices, dtype = cp.int32)
    rowptr = cp.array(datah.indptr, dtype = cp.int32)
    
    data = cpsp.csr_matrix((v, idx, rowptr))
    del v
    del idx
    del rowptr

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
  #Xh.indptr = np.pad(Xh.indptr, pad_width, "edge")
   
  X = cpsp.csr_matrix((X.data, X.indices, X.indptr))
  #Xh = sp.csr_matrix((Xh.data, Xh.indices, Xh.indptr))
  #X = cp.pad(X, (0,diff), "constant")
  n, dim = X.shape
  

ppl = int(n/leaves)


print('Number of poitns =', n, ', and the dimension =', dim)
print('Tree depth =', depth)
print('pts/leaf =', ppl)
print('Warning depth<=dim, will use non-orthogonal directions')

#nbrs = NearestNeighbors(n_neighbors=K,algorithm='brute').fit(X)
print('Starting tree iteration')

fname = '../results/' + dataset + '_ex/'
knnidx_ex = np.array(np.load(fname + 'knnId_ex.npy'), dtype = np.int32)
knndis_ex = np.array(np.load(fname + 'knnDist_ex.npy'), dtype = np.float32)
test_pt = np.array(np.load(fname + 'test_pt.npy'), dtype = np.int32)
knndis_ex = np.nan_to_num(knndis_ex)



#monitor(0, knnidx, knndis)

tic = time.time();
'''
gids = cp.arange(0, n, dtype = cp.int32)
#for i in range(leaves):
#  gids[i*ppl:(i+1)*ppl] = cp.random.permutation(gids[i*ppl:(i+1)*ppl])

knnidx, knndis = py_sfiknn(gids, X, leaves, K, knndis, knnidx)
'''
mempool = cp.get_default_memory_pool()
mempool.free_all_blocks()
mempool = cp.get_default_memory_pool()
print(" %.4f from %.4f is occupied "%(mempool.used_bytes()/1e9, mempool.total_bytes()/1e9))

knnidx, knndis = rt.rkdt_a2a_it(X,depth,knnidx, knndis, K,T,monitor,0, False)
#knnidx, knndis = rt.rkdt_a2a_it(X,depth,knnidx, knndis, K,T,None,0, False)
toc = time.time();
print(knndis_ex[0:5, :])
print(knnidx_ex[0:5, :])
print(test_pt[0:5])
print(knndis[test_pt[0:5], :])
print(knnidx[test_pt[0:5], :])


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
