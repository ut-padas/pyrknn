from prknn.kdforest.mpi.tree import *
from prknn.kdforest.mpi.util import *
from prknn.kdforest.mpi.forest import *

from mpi4py import MPI

import numpy as np

import time
import platform

from sklearn.datasets import fetch_openml
from sklearn.preprocessing import normalize

sparse = True

def test_build():

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    N = 2**19
    d= 10000
    k= 64
    np.random.seed(10)

    if sparse:

        if os.path.exists('test_sparse_data.bin'):
            data = np.fromfile("test_sparse_data.bin", dtype=np.float32)
            indptr = np.fromfile("test_sparse_ptr.bin", dtype=np.int32)
            idx = np.fromfile("test_sparse_idx.bin", dtype=np.int32)
        else:
            g_arr = sp.random(N, d, density=0.0001, format='csr', dtype=np.float32)
            data = np.array(g_arr.data, dtype=np.float32)
            indptr = np.array(g_arr.data, dtype=np.int32)
            idx = np.array(g_arr,data, dtype=np.int32)

            data.tofile("test_sparse_data.bin")
            indptr.tofile("test_sparse_ptr.bin")
            idx.tofile("test_sparse_idx.bin")

        print("... completed making dataset")
    else:
        g_arr = np.random.rand(N, d)
        g_arr = np.array(g_arr, dtype=np.float32)

    #Grab local portion (strong scaling)
    nlocal = N/size
    local_idx = np.arange(rank*nlocal, (rank+1)*nlocal, dtype=np.int32)
    local_arr = g_arr[local_idx]

    if sparse:
        A = local_arr.tocoo()
    else:
        A = np.copy(local_arr) 

    tree = RKDForest(pointset=A, levels=20, leafsize=2048, comm=comm, location="GPU", ntrees=1, sparse=sparse)
    approx = tree.aknn_all_build(k, ntrees=1, blockleaf=10, blocksize=64, cores=56)

    print("Results", rank, approx, flush=True)

    #if rank == 0:
    #    print("0 is ", local_arr[0])
    #    print("0th neighbors", g_arr[approx[0][0, :]])

total_t = time.time()
g = test_build()
total_t = time.time() - total_t
print("Total t", total_t)
