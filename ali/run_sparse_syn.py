#from time import time
#import rkdt as rt
import sys
from sklearn.neighbors import NearestNeighbors
from sklearn.datasets import load_svmlight_file
sys.path.append("..")

import time
import platform 
import os 
from joblib import Memory
import argparse
import numpy as np
import cupy as cp
import scipy.sparse as sp
import cupy.random 
import cupyx.scipy.sparse as cpsp
from datetime import datetime
from scipy.sparse import csr_matrix
import scipy.sparse as sp
from utils.gensparse import gen_random_sparse_csr
from src.sparse.seqsearch_full.queryknn_seqsearch import *
from src.sparse.seqsearch_fused.queryknn_seqsearch import *

from src.sparse.guided_full.queryknn_guided import *
from src.sparse.guided_full_nodatacopy.queryknn_guided import *
from src.sparse.guided_full_copydata.queryknn_guided import *
from utils.queryknn import queriesleafknn


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
leaves = 1 << depth

DAT_DIR='/scratch/07544/ghafouri/datasets'

if dataset == 'url':
  X = get_url_data()  
  v = cp.array(X.data, dtype = cp.float32)
  idx = cp.array(X.indices, dtype = cp.int32)
  rowptr = cp.array(X.indptr, dtype = cp.int32)
  X = cpsp.csr_matrix((v, idx, rowptr))
  del v
  del idx
  del rowptr
  n, dim = X.shape
  N = n 
  ppl = int(cp.ceil(n / leaves))
  n_true = cp.array(ppl * leaves)
  diff = n_true - N
  last = cp.asarray(X.indptr[N])
  pad_width = np.zeros(2, dtype=np.int32)
  end_val = cp.zeros(2, dtype=cp.int32)
  pad_width[1] = diff
  end_val[1] = last

  if diff > 0:
    X.indptr = cp.pad(X.indptr, pad_width, "edge") 
    X = cpsp.csr_matrix((X.data, X.indices, X.indptr))
    n, dim = X.shape
  



  n,dim = X.shape
elif dataset == 'avazu':
  X = get_avazu_data()
  n,dim = X.shape
elif dataset == 'syn':
  '''
  path = '/scratch/07544/ghafouri/pyrknn/GeMM/pysrc/dataset/syn/'
  testcase = 'syn_n16_sp9990_nq1024_k64/'
  X = sp.load_npz(path + testcase + 'data.npz')
  X = cpsp.csr_matrix((cp.asarray(X.data), cp.asarray(X.indices), cp.asarray(X.indptr)))
  test_pt = np.load(path + testcase + 'test_pt.npy')
  knndis_ex = np.load(path + testcase + 'dis_ex.npy')
  knnidx_ex = np.load(path + testcase + 'id_ex.npy')
  n,dim = X.shape 
  '''
  X = gen_random_sparse_csr(n,dim,avgnnz)
  #X = sp.load_npz("/scratch/07544/ghafouri/pyrknn/GeMM/pysrc/compare/full/data.npz")
  X = cpsp.csr_matrix((cp.asarray(X.data, dtype=cp.float32), cp.asarray(X.indices, dtype=cp.int32), cp.asarray(X.indptr, dtype=cp.int32)))
  n,dim = X.shape
  print("n=%d, dim=%d"%(n,dim))  

print("Done !")


nq = 10000

seed = int(time.time())
cp.random.seed(seed)
pt_ind = cp.random.randint(0, n, nq, dtype=cp.int32)

X_q = X[pt_ind, :]


leafIds = cp.random.randint(0, leaves, nq, dtype=cp.int32) 
#leafIds = cp.ones(nq, dtype=cp.int32) 


 
knndis = 1e30*cp.ones((nq,K), dtype = cp.float32)
knnidx = -cp.ones((nq,K), dtype = cp.int32)         

#knnidx, knndis = py_queryknn(X, X_q, leaves, ppl, K, knndis, knnidx, 0, 1, leafIds, num_search_leaves)

print("seq full")
tic = time.time()
knnidx, knndis , qId = py_queryknn_seqsearch(X, X_q, leaves, ppl, K, knndis, knnidx, 0, 1, leafIds)
toc = time.time() - tic
print("taks %.4f sec"%toc)

print("qId = %d, leaf = %d"%(qId, leafIds[qId]))

tic = time.time()
knnidx_ex, knndis_ex = queriesleafknn(X, X_q, leaves, ppl, K, leafIds, qId)
toc = time.time() - tic 
print("Exact for one point takes %.4f sec"%toc)

er = cp.linalg.norm(knndis[qId, :] - knndis_ex)
print("err = %.4f"%er)


knndis = 1e30*cp.ones((nq,K), dtype = cp.float32)
knnidx = -cp.ones((nq,K), dtype = cp.int32)         

print("seq fused")
tic = time.time()
knnidx, knndis , qId = py_queryknn_seqsearch_fused(X, X_q, leaves, ppl, K, knndis, knnidx, 0, 1, leafIds)
toc = time.time() - tic
print("taks %.4f sec"%toc)


print("qId = %d, leaf = %d"%(qId, leafIds[qId]))

tic = time.time()
knnidx_ex, knndis_ex = queriesleafknn(X, X_q, leaves, ppl, K, leafIds, qId)
toc = time.time() - tic 
print("Exact for one point takes %.4f sec"%toc)


er = cp.linalg.norm(knndis[qId, :] - knndis_ex)
print("\nerr = %.4f\n"%er)



knndis = 1e30*cp.ones((nq,K), dtype = cp.float32)
knnidx = -cp.ones((nq,K), dtype = cp.int32)         

print("guided full")
tic = time.time()
knnidx, knndis , qId = py_queryknn_guided(X, X_q, leaves, ppl, K, knndis, knnidx, 0, 1, leafIds)
toc = time.time() - tic
print("taks %.4f sec"%toc)


tic = time.time()
knnidx_ex, knndis_ex = queriesleafknn(X, X_q, leaves, ppl, K, leafIds, qId)
toc = time.time() - tic 
print("Exact for one point takes %.4f sec"%toc)


er = cp.linalg.norm(knndis[qId, :] - knndis_ex)
print("err = %.4f"%er)


print(knnidx[qId, :])
print(knndis[qId, :])

print(knnidx_ex)
print(knndis_ex)




'''
print(knnidx[qId, :])
print(knndis[qId, :])

print(knnidx_ex)
print(knndis_ex)




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
pad_width = cp.zeros(2, dtype=cp.int32)
end_val = cp.zeros(2, dtype=cp.int32)
pad_width[1] = diff
end_val[1] = last

if diff > 0:
  X.indptr = cp.pad(X.indptr, pad_width, "edge") 
  X = cpsp.csr_matrix((X.data, X.indices, X.indptr))
  n, dim = X.shape
  

ppl = int(n/leaves)


print('Number of poitns =', n, ', and the dimension =', dim)
print('Tree depth =', depth)
print('pts/leaf =', ppl)
print('Warning depth<=dim, will use non-orthogonal directions')

#nbrs = NearestNeighbors(n_neighbors=K,algorithm='brute').fit(X)
print('Starting tree iteration')

if dataset == 'url' or dataset == 'avazu':
  fname = '../results/' + dataset + '_ex/'
  knnidx_ex = np.array(np.load(fname + 'knnId_ex.npy'), dtype = np.int32)
  knndis_ex = np.array(np.load(fname + 'knnDist_ex.npy'), dtype = np.float32)
  test_pt = np.array(np.load(fname + 'test_pt.npy'), dtype = np.int32)
  knndis_ex = np.nan_to_num(knndis_ex)



#monitor(0, knnidx, knndis)

tic = time.time();
gids = cp.arange(0, n, dtype = cp.int32)
knnidx, knndis = py_sfiknn(gids, X, leaves, K, knndis, knnidx, 0)

mempool = cp.get_default_memory_pool()
mempool.free_all_blocks()
mempool = cp.get_default_memory_pool()
print(" %.4f from %.4f is occupied "%(mempool.used_bytes()/1e9, mempool.total_bytes()/1e9))

#knnidx, knndis = rt.rkdt_a2a_it(X,depth,knnidx, knndis, K,T,monitor,0, False)
#knnidx, knndis = rt.rkdt_a2a_it(X,depth,knnidx, knndis, K,T,None,0, False)
toc = time.time();

print('RKDT took', '{:.2f}'.format(toc-tic), 'secs \n')

#cp.save("results/sparse/knndis", knndis)
#cp.save("results/sparse/knnidx", knnidx)

#knndis = cp.load("../results/sparse/knndis.npy")
#knnidx = cp.load("../results/sparse/knnidx.npy")



print("computing the exact neghobors")
#test_pt = cp.arange(0, 1000)
'''
