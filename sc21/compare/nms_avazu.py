import numpy as np
import sys
import nmslib
import time
import math
from scipy.sparse import vstack as sparse_stack
import os

import argparse

from sklearn.datasets import load_svmlight_file
from joblib import Memory

from pyrknn.kdforest.mpi.util import *
set_env("CPU", True)

def read_result(filename, m, k):
    with open('ref_avazu_full.txt') as f:
        flat_list=[word for line in f for word in line.split()]

    flat_list.reverse()
    print("Length: ", len(flat_list))

    nborID = np.zeros((m, k), dtype=np.int32)
    nborDist = np.zeros((m, k), dtype=np.float32)
    m = int(flat_list.pop())
    k = int(flat_list.pop())
    print(m, k)
    for i in range(m):
        for j in range(k):
            nborID[i, j] = int(flat_list.pop())


    for i in range(m):
        for j in range(k):
            nborDist[i, j] = float(flat_list.pop())
            
    return (nborID, nborDist)


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
    data_app = load_svmlight_file(os.environ["SCRATCH"]+"/datasets/avazu/avazu-app", n_features=1000000)
    data_site = load_svmlight_file(os.environ["SCRATCH"]+"/datasets/avazu/avazu-site", n_features=1000000)
    print(data_app[0], data_app[1])
    print(data_app[0].shape, data_app[1].shape)
    t = time.time() - t
    print("It took ", t, " (s) to load the dataset")
    return data_app[0], data_site[0]

print("Starting to Read Data", flush=True)
X_app, X_site = get_data()
print(X_app.shape, X_site.shape)
X = sparse_stack([X_app, X_site])

N, d = X.shape
print(X)

print("Finished Reading Data", flush=True)
k = 64
N  = X.shape[0]
d  = X.shape[1]
print("Data shape: ", (N, d))
truth = read_result("ref_avazu_full.txt", 100, 64)


data_matrix = X.tocsr()



print("Data Matrix Size is: ", N, d, flush=True)

#Compute Exact Solution

#Convert Q to the correct datatype

parser = argparse.ArgumentParser(description='Test HNSW parameters')

parser.add_argument('-M', metavar='M', type=int, default=50)
parser.add_argument('-efC', metavar='efC', type=int, default=100)
parser.add_argument('-efS', metavar='efS', type=int, default=100)
parser.add_argument('-post', metavar='post', type=int, default=0)
parser.add_argument('-type', metavar='type', type=int, default=2)

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

    approx = (a[:nq], b[:nq])

    hit_rate, rel_err, mean_sim = accuracy_check(truth, approx)
    print(f"Search: {M}, {efC}, {efS}, {search_t*(N/batch)}, {hit_rate}, {rel_err}, {mean_sim}")
    #print(n/batch)
    if hit_rate > 0.99:
        flag = False 
    if efS > 1000:
        flag = False



