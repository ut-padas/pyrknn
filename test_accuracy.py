from pyrknn.kdforest.mpi.tree import *
from pyrknn.kdforest.mpi.util import *
from pyrknn.kdforest.mpi.forest import *

from mpi4py import MPI

import numpy as np

import time
import platform

from sklearn.datasets import fetch_openml
from sklearn.preprocessing import normalize

weak = False
sparse = False

def test_build():

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    d = 10
    N = 2**17
    data = None
    filename = os.environ["SCRATCH"]+'/datasets/sphere_50/sphere_set_'+str(rank)+'.bin'

    tdata = np.fromfile(filename, dtype=np.float32)
    tdata = np.reshape(tdata, (len(tdata)//d, d))
    if data is None:
        data = tdata

    data = data[:N]


    filename = os.environ["SCRATCH"]+'/datasets/sphere_50/sphere_set_'+str(0)+'.bin'
    tdata = np.fromfile(filename, dtype=np.float32)
    tdata = np.reshape(tdata, (len(tdata)//d, d))

    data = np.random.rand(N, d)
    data = np.asarray(data, dtype=np.float32)
    tdata = data

    k = 32
 
    nq = 1000
    Q = tdata[:nq, ...]

    A = np.copy(data) 
    tree = RKDForest(pointset=A, levels=0, leafsize=256, location="CPU", ntrees=1, sparse=sparse)
    tree.build()
    tree = tree.forestlist[0]
    truth = tree.dist_exact(Q, k)

    #truth = (np.zeros([nq, k], dtype=np.int32), np.zeros([nq, k], dtype=np.float32))

    tree = RKDForest(pointset=data, levels=20, leafsize=256, comm=comm, location="CPU", ntrees=5, sparse=sparse)
    approx = tree.aknn_all_build(k, ntrees=2, blockleaf=2, blocksize=32, cores=56, truth=truth, until=True)

    """
    while True:
        try:
            approx = tree.aknn_all_build(k, ntrees=2, blockleaf=2, blocksize=32, cores=4, truth=truth, until=False)
            break
        except:
            pass
    
    """
    if rank == 0:
        print("Final Solution", rank, approx, flush=True)
        print("Truth", rank, truth, flush=True)
        print(Primitives.timing())

total_t = time.time()
test_build()
total_t = time.time() - total_t
Primitives.timing()
