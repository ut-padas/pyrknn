from pyrknn.kdforest.mpi.util import *
from pyrknn.kdforest.mpi.forest import *

import numpy as np
import time

N = 2**21
d = 10
k = 64
levels=10
blocksize=32
ntrees = 2

np.random.seed(10)
X = np.random.rand(N, d)
Z = np.array(X, dtype=np.float32)
device = 0

gids = np.arange(N, dtype=np.int32)
t = time.time()
approx = gpu.dense_knn(gids, Z, levels, ntrees, k, blocksize, device)
t = time.time() - t

print("Total Time", t)
print(approx)



