from pyrknn.kdforest.mpi.tree import *
from pyrknn.kdforest.mpi.util import *
from pyrknn.kdforest.mpi.forest import *

from mpi4py import MPI
import numpy as np

import time
import platform

import os

from sklearn.datasets import load_svmlight_file
from joblib import Memory

mem = Memory("./mycache")

@mem.cache()
def get_data():
    t = time.time()
    data = load_svmlight_file(os.environ["SCRATCH"]+"/pyrknn/datasets/url_combined", n_features=3231961)
    t = time.time() - t
    print("It took ", t, " (s) to load the dataset")
    return data[0], data[1]

def run():
    sparse = True

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    print("Starting to Read Data", flush=True)
    X, y = get_data()
    print("Finished Reading Data", flush=True)

    N, d = X.shape

    n_local = N//size

    #X = X[:200]
    start = (rank)*n_local
    end   = (rank+1)*n_local
    nq = 1000

    print("Starting to Preprocess Data", flush=True)

    Q = X[:nq].tocsr()

    #Convert Q to the correct datatype
    q_data = np.asarray(Q.data, dtype=np.float32)
    q_indices = np.asarray(Q.indices, dtype=np.int32)
    q_indptr = np.asarray(Q.indptr, dtype=np.int32)
    Q = sp.csr_matrix( (q_data, q_indices, q_indptr), shape=(nq, d))

    print(Q.indices)

    X = X[start:end].tocoo()

    print("Finished Preprocessing Data")

    k = 64

    #Build Forest
    forest = RKDForest(pointset=X, levels=0, leafsize=256, location="CPU", ntrees=1, sparse=sparse, N=n_local, d=d)
    forest.build()

    print("Finished Building Tree")

    t = time.time()
    #Compute true solution with brute force on nq subset
    truth = forest.dist_exact(Q, k)
    t = time.time() - t
    print("Exact Search took: ", t, " (s)")

    print("Finished Exact Search")
    print("Truth:", truth)

    nbrID = truth[0]
    nbrDist = truth[1]

    np.save("url_nborID_1000.bin", nbrID)
    np.save("url_nborDist_1000.bin", nbrDist)

run()








