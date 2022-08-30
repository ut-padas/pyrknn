import numpy as np
import sys
import nmslib
import time
import math
from scipy.sparse import vstack as sparse_stack
import os

import scipy.sparse as spl

import argparse

from sklearn.datasets import load_svmlight_file
from joblib import Memory

from pyrknn.kdforest.util import *
from pyrknn.kdforest.tree import *
from pyrknn.kdforest.forest import *

set_env("CPU", True)

print("Starting Script", flush=True)
mem = Memory("./mycache")
name = os.environ["SCRATCH"]+"/comparison/avazu/"
name = "avazu"



path='/work/06081/wlruys/frontera/workspace/scikit-build-sample-projects/projects/pyrknn_master/pyrknn/sc21/compare'

def read_truth(path):

    fpath_id = f"/url_id_${k}_${nq}.npy"
    fpath_dist = f"/url_dist_${k}_${nq}.npy"

    ID = np.load(path+fpath_id)
    DIST = np.load(path+fpath_dist)

    return (ID, DIST)

@mem.cache()
def get_data():
    t = time.time()
    data = load_svmlight_file(path+"/url_combined", n_features=3231961)
    t = time.time() - t
    print("It took ", t, " (s) to load the dataset")
    return data[0]

print("Starting to Read Data", flush=True)
X = get_data()
#print(X_app.shape, X_site.shape)
#X = sparse_stack([X_app, X_site])

N, d = X.shape
print(X)

print("Finished Reading Data", flush=True)
k = 64
N  = X.shape[0]
d  = X.shape[1]
print("Data shape: ", (N, d))


nq = 100
X = X.tocsr()
Q = X[:nq]
q_data = np.asarray(Q.data, dtype=np.float32)
q_indices = np.asarray(Q.indices, dtype=np.int32)
q_indptr = np.asarray(Q.indptr, dtype=np.int32)
Q = spl.csr_matrix( (q_data, q_indices, q_indptr), shape=(nq, d))


data_matrix = X


t = time.time()

timer = Profiler()
record = Recorder()

#Compute true solution with brute force on nq subset
#C = X.copy()

fpath_id = f"/url_id_${k}_${nq}.npy"
fpath_dist = f"/url_dist_${k}_${nq}.npy"

if not os.path.isfile(path+fpath_id):
    tree = RKDT(data=X, levels=0, leafsize=2048, location="HOST")
    truth = tree.distributed_exact(Q, k)
    truth = merge_neighbors(truth, truth, k)

    np.save(path+fpath_id, truth[0])
    np.save(path+fpath_dist, truth[1])
else:
    truth = read_truth(path);
    truth = merge_neighbors(truth, truth, k)
t = time.time() - t

print("Exact Search took: ", t, " (s)", flush=True)



print("Data Matrix Size is: ", N, d, flush=True)

#Compute Exact Solution

#Convert Q to the correct datatype

parser = argparse.ArgumentParser(description='Test HNSW parameters')

parser.add_argument('-M', metavar='M', type=int, default=50)
parser.add_argument('-efC', metavar='efC', type=int, default=50)
parser.add_argument('-efS', metavar='efS', type=int, default=100)
parser.add_argument('-post', metavar='post', type=int, default=1)
parser.add_argument('-type', metavar='type', type=int, default=0)

args = parser.parse_args()


nq = truth[0].shape[0]

#Compute Approximate Solution

M  = args.M
efC = args.efC

num_threads = 56
index_time_params = {'delaunay_type':args.type, 'M': M, 'indexThreadQty':num_threads, 'efConstruction' : efC, 'post' : args.post}


t = time.perf_counter()
index = nmslib.init(method='hnsw', space='l2_sparse', data_type=nmslib.DataType.SPARSE_VECTOR)
index.addDataPointBatch(data_matrix)
index.createIndex(index_time_params)
index_t = time.perf_counter() - t
print(f"Index: {M}, {efC}, {index_t}")

efS = args.efS
batch = 60000

flag = True
efS = 0
while flag:
    efS = efS + 4
    query_time_params = {'efSearch': efS}
    index.setQueryTimeParams(query_time_params)

    t  = time.perf_counter()
    neighbors = index.knnQueryBatch(data_matrix[:batch], k=k, num_threads=num_threads)
    search_t = time.perf_counter() - t

    #print(neighbors)
    FLOAT_MAX = 1e30
    a, b = map(list, zip(*neighbors))
    print(a[0].shape)
    for i in range(len(a)):
        if (a[i].shape != k):
            ipad = np.zeros(k, dtype=np.int32)+-1
            dpad = np.zeros(k, dtype=np.int32)+ FLOAT_MAX
            ipad[:len(a[i])] = a[i]
            dpad[:len(a[i])] = b[i]
            a[i] = ipad
            b[i] = dpad

    a = np.stack(a, axis=0)
    b = np.stack(b, axis=0)
    print("Truth", truth, flush=True)
    approx = (a[:nq], b[:nq]**2)

    print("Approx", approx, flush=True)

    hit_rate, rel_err, mean_sim = check_accuracy(truth, approx)
    print(f"Search: {M}, {efC}, {efS}, {search_t*(N/batch)}, {hit_rate}, {rel_err}, {mean_sim}", flush=True)
    #print(n/batch)
    if hit_rate > 0.99:
        flag = False
    if efS > 1000:
        flag = False



