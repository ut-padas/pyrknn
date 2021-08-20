import numpy as np
import time
from pyrknn.kdforest.mpi.util import *
set_env("CPU", False)

from pyrknn.kdforest.mpi.tree import *
from pyrknn.kdforest.mpi.util import *
from pyrknn.kdforest.mpi.forest import *

d = 15
n = 2**22
k = 64

path = "/scratch1/06081/wlruys/datasets/sift/sift/sift_base.fvecs" 

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
X = fvecs_read(path).astype(np.float32)
X = normalize(X)
np.random.seed(10)
n, d = X.shape
xb = X
nq = 1000
gids = np.arange(n, dtype=np.int32)
q = xb[:nq]
truth = direct_knn(gids, xb, q, k)
truth = merge_neighbors(truth, truth, k)

print(X.shape)

import faiss
faiss.omp_set_num_threads(56)
res = faiss.StandardGpuResources()
t = time.time()
quantizer = faiss.IndexFlatL2(d)
index = faiss.IndexIVFFlat(quantizer, d, 2048, faiss.METRIC_L2)
print("starting index")
index = faiss.index_cpu_to_gpu(res, 0, index)

print("training index")

index.train(xb)
print(index.is_trained)
index.add(xb)
print(index.ntotal)
t = time.time() - t
print("Index Build: ", t)
index.nprobe = 47
t = time.time()
D, I = index.search(xb[:20000], k)
t = time.time() - t
print("Search: ", t)
print("Queried: ", len(xb))
print("I, D shape", I.shape, D.shape)
approx = (I[:nq], D[:nq])
result = accuracy_check(truth, approx)
print("Accuracy", result)
