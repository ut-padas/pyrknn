import numpy
import sys
import nmslib
import time
import math
import scipy.sparse as sp
import os

import argparse

from sklearn.datasets import load_svmlight_file
from joblib import Memory

from pyrknn.kdforest.util import *
from sklearn.preprocessing import normalize 

set_env("CPU", False)


N = (2**20)
path = "/work2/06081/wlruys/frontera/datasets/train-features.npy"
X = np.load(path)
X = X[:N]
def fvecs_read(filename, c_contiguous=True):
    fv = np.fromfile(filename, dtype=np.float32)
    if fv.size == 0:
        return np.zeros((0, 0))
    dim = fv.view(np.int32)[0]
    assert dim > 0
    fv = fv.reshape(-1, 1 + dim)
    if not all(fv.view(np.int32)[:, 0] == dim):
        raise IOError("Non-uniform vector sizes in " + filename)
    fv = fv[:, 1:]
    if c_contiguous:
        fv = fv.copy()
    return fv

nq = 100
k = 32 

#X = fvecs_read(path).astype(np.float32)
#X = normalize(X)

n, d = X.shape
xb = X

print(n, d)
gids = np.arange(n, dtype=np.int32)
q = xb[:nq]
truth = direct_knn(gids, xb, q, k)
truth = merge_neighbors(truth, truth, k)

print(truth)
#Convert Q to the correct datatype

parser = argparse.ArgumentParser(description='Test HNSW parameters')

parser.add_argument('-M', metavar='M', type=int, default=32)
parser.add_argument('-efC', metavar='efC', type=int, default=60)
parser.add_argument('-efS', metavar='efS', type=int, default=100)
parser.add_argument('-post', metavar='post', type=int, default=2)
parser.add_argument('-type', metavar='type', type=int, default=0)

args = parser.parse_args()

#Compute Approximate Solution

M  = args.M
efC = args.efC

num_threads = 56
index_time_params = {'delaunay_type':args.type, 'M': M, 'indexThreadQty':num_threads, 'efConstruction' : efC, 'post' : args.post}

index = nmslib.init(method='hnsw', space='l2')
index.addDataPointBatch(xb)

t = time.perf_counter()
index.createIndex(index_time_params, print_progress=True)
index_t = time.perf_counter() - t
print(f"Index: {M}, {efC}, {index_t}")

batch = 20000

flag = True 
efS = 0
while flag:
    efS = efS + 1
    query_time_params = {'efSearch': efS}
    index.setQueryTimeParams(query_time_params)

    t  = time.perf_counter()
    neighbors = index.knnQueryBatch(xb[:batch], k=k, num_threads=num_threads)
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

    approx = (a[:nq], b[:nq])

    hit_rate, rel_err, mean_sim = check_accuracy(truth, approx)
    print(f"Search: {M}, {efC}, {efS}, {search_t*(n/batch)}, {hit_rate}, {rel_err}, {mean_sim}")

    if hit_rate > 0.99:
        flag = False 
    if efS > 1000:
        flag = False

