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
import sys
sys.path.append("../../wrapper")
from cuda_wrapper.sparse import *
#import cupy as cp

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
    prefix = "/scratch/07544/ghafouri"
    data_app = load_svmlight_file(prefix + "/datasets/avazu/avazu-app", n_features=1000000, dtype=np.float32)
    data_site = load_svmlight_file(prefix+"/datasets/avazu/avazu-site", n_features=1000000, dtype=np.float32)
    print(data_app[0], data_app[1])
    print(data_app[0].shape, data_app[1].shape)
    t = time.time() - t
    print("It took ", t, " (s) to load the dataset")
    return data_app[0], data_site[0]

@mem.cache()
def get_url_data():
    t = time.time()
    data = load_svmlight_file(os.environ["SCRATCH"]+"/datasets/url/url_combined", n_features=3231961)
    t = time.time() - t
    print("It took ", t, " (s) to load the dataset")
    return data[0]




'''
print("Starting to Read Data", flush=True)
X_app, X_site = get_data()
print(X_app.shape, X_site.shape)
X = sparse_stack([X_app, X_site])
'''
X = get_url_data()


print("Finished Reading Data", flush=True)
K = args.k
depth = args.levels
#X = X[:n,]

N  = X.shape[0]
d  = X.shape[1]
X.sort_indices()
print("Init Data shape: ", (N, d))

#truth = read_truth(name, k)


local_size = N//size
#X = X[rank*local_size:(rank+1)*local_size]

#timer = Profiler()
#record = Recorder()

np.random.seed(args.seed)





def apknnerr( ex_id,ex_dist, ap_id,ap_dist ,nc):
     
    '''
    rowerr = np.any(ex[:nc,]-ap[:nc,],axis=1)
    rowidx = np.where(rowerr==True)
    acc = 1 - len(rowidx[0])/nc
    '''
    err = 0.0
    for i in range(nc):
        miss_array_id = [1 if ap_id[i, j] in ex_id[i, :] else 0 for j in range(K)]
        miss_array_dist = [1 if ap_dist[i, j] <= ex_dist[i, -1] else 0 for j in range(K)]
        #err += np.sum(np.logical_or(miss_array_id, miss_array_dist))
        err += np.sum(miss_array_id)
    acc = err/(nc*K)


    return acc

def apknnerr_dis(ex,ap,nc):
    err =np.linalg.norm(ex[:nc,]-ap[:nc,])/np.linalg.norm(ex[:nc,])
    print(err)
    return err
    
                         

def monitor(t,knnidx,knndis):
    tol = 0.95
    acc = apknnerr(knnidx_ex,knndis_ex, knnidx, knndis,nex)
    derr =apknnerr_dis(knndis_ex,knndis,nex)
    #derr = cp.asnumpy(derr)
    cost = t*points_per_leaf
    print('it = ', '{:3d}'.format(t), 'Recall accuracy:', '{:.4f}'.format(acc), 'distance error = {:.4f}'.format(derr), 'cost = %.4f'%cost)
    break_iter = False
    break_iter =  (acc>tol or cost>n)
    return break_iter


print('Padding the data')




T = args.iter
dim = X.shape[1]
n = X.shape[0]

leaves = 1 << depth 
ppl = np.ceil(n / leaves)
n_true = int(ppl * leaves)
diff = n_true - n
last = X.indptr[n]
avgnnz = np.mean(np.diff(X.indptr))
print("avgnnz = %d "%avgnnz)
if diff > 0:
  X = csr_matrix((X.data, X.indices, np.pad(X.indptr, (0, diff), "linear_ramp", end_values=(0,last))))
  n, dim = X.shape

points_per_leaf = int(n/leaves)






print('Number of poitns =', n, ', and the dimension =', dim)
print('Tree depth =', depth)
print('points_per_leaf =', points_per_leaf)
print('Warning depth<=dim, will use non-orthogonal directions')

indptr = X.indptr
diff = np.diff(indptr)

nex = points_per_leaf
t = 0
#nbrs = NearestNeighbors(n_neighbors=K,algorithm='brute').fit(X)
fname = "url_ex/url_knnidx_ex_L%d.npy"%depth
print(fname)

if os.path.isfile(fname):
  print("loading the neighbors")
  knndis_ex = np.load("url_ex/url_knndis_ex_L%d.npy"%depth)
  knnidx_ex = np.load("url_ex/url_knnidx_ex_L%d.npy"%depth)
else:
  print("computing the exact neghobors")
  nbrs = NearestNeighbors(n_neighbors=K,algorithm='brute').fit(X[:nex,])
  knndis_ex, knnidx_ex = nbrs.kneighbors(X[:nex,])
  knndis_ex = np.asarray(knndis_ex)
  knnidx_ex = np.asarray(knnidx_ex)
  np.save("url_ex/url_knndis_ex_L%d"%depth, knndis_ex)
  np.save("url_ex/url_knnidx_ex_L%d"%depth, knnidx_ex)


knndis = np.ones((n,K), dtype = np.float32) + 1e30
knnidx = -np.ones((n,K), dtype = np.int32)         
#gids = np.zeros(n).astype('int32')
gids = np.arange(n, dtype = np.int32);
#gids = np.random.permutation(np.arange(n, dtype = np.int32))
print('Starting tree iteration')

tic = time.time();
leaves = int(n // points_per_leaf)

#knnidx, knndis = py_sfiknn(gids, X, leaves, K, knndis.ravel(), knnidx.ravel(), 0)
#knnidx, knndis = rt.rkdt_a2a_it(X,gids,depth,knndis.ravel(), knnidx.ravel(), K,T,monitor,0, False)
knnidx, knndis = rt.rkdt_a2a_it(X,gids,depth,knnidx, knndis, K,T,monitor,1, False)
toc = time.time();
print('RKDT took', '{:.2f}'.format(toc-tic), 'secs')
monitor(0,knnidx,knndis)

err = np.absolute(knnidx_ex - knnidx[:nex,])
err_o = err[err > 0]
acc = 1 - err_o.shape[0]/(err.flatten().shape[0])
err_val = np.linalg.norm((knndis_ex - knndis[:nex,]).flatten())
print('Recall accuracy:', '{:.4f}'.format(acc))
print('Recall error val:', '{:.4f}'.format(err_val))


'''
print(knnidx_ex[0, :])
print(knnidx[0, :])
print(knndis_ex[0, :])
print(knndis[0, :])
print(knnidx_ex[0, :] - knnidx[0, :])

for i in range(nex):	
    if not np.array_equal(knnidx_ex[i,:], knnidx[t*nex + i,:]):
      print(i)
      print('ex')
      print(knnidx_ex[i,:])
      print(knndis_ex[i, :])
      print('rec')
      print(knnidx[t*nex + i,:])
      print(knndis[t*nex+i, :])
      print(knnidx_ex[i,:] - knnidx[t*nex + i,:])
'''
      
if rank == 0:
  timer.print()
  record.print()