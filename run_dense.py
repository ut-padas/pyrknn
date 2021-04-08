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

from pyrknn.kdforest.mpi.util import *
set_env("CPU", False)

print("Starting to Read Data", flush=True)

N = 2**22
d = 15

np.random.seed(1001)
data_matrix = np.random.randn(N, d)
data_matrix = np.asarray(data_matrix, dtype=np.float32)
print("Data Matrix Size is: ", N, d, flush=True)

k = 64

#Compute Exact Solution
nq = 1000
gids = np.arange(N, dtype=np.int32)
q = data_matrix[:nq]
truth = single_knn(gids, data_matrix, q, k)
truth = merge_neighbors(truth, truth, k)

#Convert Q to the correct datatype

parser = argparse.ArgumentParser(description='Test HNSW parameters')

#Default
#parser.add_argument('-M', metavar='M', type=int, default=50)
#parser.add_argument('-efC', metavar='efC', type=int, default=100)
#parser.add_argument('-efS', metavar='efS', type=int, default=100)
#parser.add_argument('-post', metavar='post', type=int, default=0)
#parser.add_argument('-type', metavar='type', type=int, default=2)

parser.add_argument('-M', metavar='M', type=int, default=20)
parser.add_argument('-efC', metavar='efC', type=int, default=50)
parser.add_argument('-efS', metavar='efS', type=int, default=50)
parser.add_argument('-post', metavar='post', type=int, default=1)
parser.add_argument('-type', metavar='type', type=int, default=0)

args = parser.parse_args()

#Compute Approximate Solution

M  = args.M
efC = args.efC

num_threads = 56
index_time_params = {'delaunay_type':args.type, 'M': M, 'indexThreadQty':num_threads, 'efConstruction' : efC, 'post' : args.post}

index = nmslib.init(method='hnsw', space='l2')
index.addDataPointBatch(data_matrix)

start = time.time()
index.createIndex(index_time_params, print_progress=True)
end = time.time()

print('Indexing time = ', (end-start) )

efS = args.efS
efS = efC

query_time_params = {'efSearch': efS}
index.setQueryTimeParams(query_time_params)

start  = time.time()
neighbors = index.knnQueryBatch(data_matrix, k=k, num_threads=num_threads)
end = time.time()

print("Query Time = ", end - start)
print("Queried ", len(neighbors), " points")

#print(neighbors)
FLOAT_MAX = 1e30
a, b = map(list, zip(*neighbors))
print(len(a))
for i in range(len(a)):
    if (a[i].shape != a[0].shape):
        ipad = np.zeros(a[0].shape, dtype=np.int32)+-1
        dpad = np.zeros(a[0].shape, dtype=np.int32)+ FLOAT_MAX
        ipad[:len(a[i])] = a[i]
        dpad[:len(a[i])] = b[i]
        a[i] = ipad
        b[i] = dpad

a = np.stack(a, axis=0)
b = np.stack(b, axis=0)

approx = (a[:nq], b[:nq])

result = neighbor_dist(truth, approx)
print("Accuracy", result)

print("result: ", result)
print("truth: ", truth)

