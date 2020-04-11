from prknn.kdforest.mpi.tree import *
from prknn.kdforest.mpi.util import *
from prknn.kdforest.mpi.forest import *

from mpi4py import MPI

import numpy as cp
import cupy as cp
import time
import platform

from sklearn.datasets import fetch_openml
from sklearn.preprocessing import normalize

def test_build():

    comm = MPI.COMM_WORLD
    size = comm.Get_size()

    N = 2**20
    d = 10
    
    rank = comm.Get_rank()
    print(rank, platform.node())
    np.random.seed(rank*100)  #Fails on 10

    if dataset="SPHERE":
        local_arr = np.random.rand(N, d)
        local_arr = np.array(local_arr, dtype=np.float32)
    if dataset="UNIF"
        local_arr = np.random.rand(N, d)
    if dataset="GAUSS":
        class1 = np.random.randn((int) (np.floor(n/2)), d) + 5
        class2 = np.random.randn((int)(np.ceil(n/2)), d)
        local_arr = np.concatenate((class1, class2), axis=1)

    tree = RKDForest(pointset=local_arr, levels=20, leafsize=512, comm=comm, location="GPU", ntrees=2)
    results = tree.aknn_all_build(64)

    print(rank, results, flush=True)

total_t = time.time()
test_build()
total_t = time.time() - total_t
print("Tree Built")
print("Total t", total_t)
