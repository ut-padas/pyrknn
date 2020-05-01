from prknn.kdforest.mpi.tree import *
from prknn.kdforest.mpi.util import *
from prknn.kdforest.mpi.forest import *

from mpi4py import MPI

import numpy as cp
import cupy as cp
import time
import platform

#from sklearn.datasets import fetch_openml
#from sklearn.preprocessing import normalize

def test_build():
    comm = MPI.COMM_WORLD
    size = comm.Get_size()

    N = 2**10
    n_local = N//size

    d = 100000
    k = 4

    dataset = False
    rank = comm.Get_rank()
    print(rank, platform.node())
    #np.random.seed(rank*10)  #Fails on 10
    if dataset:
        loadall = False
    else:
        if os.path.exists('test_sparse_data2.bin'):
            data = np.fromfile("test_sparse_data2.bin", dtype=np.float32)
            indptr = np.fromfile("test_sparse_ptr2.bin", dtype=np.int32)
            idx = np.fromfile("test_sparse_idx2.bin", dtype=np.int32)
            g_arr = sp.csr_matrix( (data, idx, indptr) , shape=(N, d));
        else:
            g_arr = sp.random(N, d, density=0.001, format='csr', dtype=np.float32)
            data = np.array(g_arr.data, dtype=np.float32)
            indptr = np.array(g_arr.indptr, dtype=np.int32)
            idx = np.array(g_arr.indices, dtype=np.int32)

            data.tofile("test_sparse_data2.bin")
            indptr.tofile("test_sparse_ptr2.bin")
            idx.tofile("test_sparse_idx2.bin")

    #print(g_arr.tocoo())
    #print(g_arr.toarray())
    
    g_c = g_arr[:N]
    #g_c = g_c.tolil()
    
    X = g_c[(rank)*n_local:(rank+1)*n_local].tocoo()

    print(rank, "Split Array", X.toarray())

    #forest = RKDForest(pointset=X, levels=20, leafsize=2048, comm=comm, location="GPU", ntrees=1, sparse=True, N=n_local, d=d)
    #forest.build()
    #tree = forest.forestlist[0]

    #print(rank, "Original Matrix", g_c.toarray())
    #for tree in forest:
    #     output = tree.data
    #     print(rank, "output", output.toarray(), tree.host_real_gids)
    #
    #     data = tree.data
    #     print(np.sum(g_c[tree.host_real_gids] != data), d*len(tree.host_real_gids))
        
    #Q = g_c[:100]
    #truth = tree.dist_exact(Q, k)
    #print(truth)
    truth = (np.zeros([100, k], dtype=np.int32), np.zeros([100, k], dtype=np.float32))

    tree = RKDForest(pointset=X, levels=20, leafsize=2048, comm=comm, location="GPU", ntrees=5, sparse=True, N=n_local, d=d)
    
    results = tree.aknn_all_build(k, ntrees=1, blockleaf=512, blocksize=16, truth=truth)

    print("Final Solution", rank, results, flush=True)

total_t = time.time()
test_build()
total_t = time.time() - total_t
print("Tree Built")
print("Total t", total_t)
