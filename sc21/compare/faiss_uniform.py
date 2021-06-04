from pyrknn.kdforest.mpi.util import * 

import numpy as np 
import time 

from sklearn.preprocessing import normalize 

set_env("CPU", False)

import argparse
import h5py




parser = argparse.ArgumentParser(description='Test HNSW parameters')

parser.add_argument('-c', metavar='c', type=int, default=2048)

args = parser.parse_args()

path = "/scratch1/06081/wlruys/orth_copy/uniform.h5"
f = h5py.File(path, 'r')
X = np.array(f["train"], dtype=np.float32)


nq = 1000
k = 32 

#X = fvecs_read(path).astype(np.float32)
#X = normalize(X)

n, d = X.shape
xb = X
gids = np.arange(n, dtype=np.int32)
q = xb[:nq]
truth = direct_knn(gids, xb, q, k)
truth = merge_neighbors(truth, truth, k)


import faiss 

faiss.omp_set_num_threads(56)
t = time.perf_counter()
clusters = args.c
quantizer = faiss.IndexFlatL2(d)
index = faiss.IndexIVFFlat(quantizer, d, clusters, faiss.METRIC_L2)
index.train(xb)
index.add(xb)
index_t = time.perf_counter()-t

print(f"Index: {clusters}, {index_t}")


batch = 20000

flag = True 
i = 0
while flag:
    i = i + 4 
    index.nprobe = i
    t = time.perf_counter()
    D, I = index.search(xb[:batch], k)
    search_t = time.perf_counter() - t

    approx = (I[:nq], D[:nq])
    hit_rate, rel_err, mean_sim = accuracy_check(truth, approx)
    print(f"Search: {clusters}, {i}, {search_t*(n/batch)}, {hit_rate}, {rel_err}, {mean_sim}")

    if hit_rate > 0.99:
        flag = False 
    if i > 1000:
        flag = False


