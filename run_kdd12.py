import cupy as cp

from pyrknn.kdforest.tree import *
from pyrknn.kdforest.util import *
from pyrknn.kdforest.forest import *

from mpi4py import MPI
import numpy as np

import time
import platform

import os

from sklearn.datasets import load_svmlight_file
from joblib import Memory

from sklearn.datasets import load_svmlight_file
from joblib import Memory
from scipy.sparse import vstack as sparse_stack
import scipy.sparse as sp
import argparse

parser = argparse.ArgumentParser(description="Test Sparse KNN")
parser.add_argument('-n', type=int, default=2**22)
parser.add_argument('-d', type=int, default=15)
parser.add_argument('-iter', type=int, default=120)
parser.add_argument('-dataset', default="kdd12")
parser.add_argument('-bs', type=int, default=64)
parser.add_argument('-bl', type=int, default=128)
parser.add_argument('-cores', type=int, default=56)
parser.add_argument('-use_gpu', type=bool, default=0)
parser.add_argument('-levels', type=int, default=18)
parser.add_argument('-k', type=int, default=32)
parser.add_argument('-leafsize', type=int, default=1024)
parser.add_argument('-ltrees', type=int, default=1)
parser.add_argument('-merge', type=int, default=1)
parser.add_argument('-overlap', type=int, default=0)
parser.add_argument('-seed', type=int, default=15)
parser.add_argument('-nq', type=int, default=1000)

args = parser.parse_args()

mem = Memory("./sparse_cache")
@mem.cache()
def get_kdd12_data():
    path = "/scratch/06081/wlruys/"
    data = load_svmlight_file(path+"kdd12", n_features=54686452)
    return data

if args.use_gpu:
    location = "GPU"
else:
    location = "HOST"

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

def run():

    if rank == 0:
        print("Starting to Read Data", flush=True)
    t_start = time.perf_counter()
    X = get_kdd12_data()
    X = X[0]
    t_end = time.perf_counter()
    if rank == 0:
        print("Finished Reading Data: ", X.shape, flush=True)
        print("It took ", t_end - t_start, " (s) to load the dataset", flush=True)

    global_n, d = X.shape

    #Grab queries
    Q = X[:args.nq]

    #Convert Q to the correct datatype
    q_data = np.asarray(Q.data, dtype=np.float32)
    q_indices = np.asarray(Q.indices, dtype=np.int32)
    q_indptr = np.asarray(Q.indptr, dtype=np.int32)
    Q = sp.csr_matrix( (q_data, q_indices, q_indptr), shape=(args.nq, d))


    #Restrict to local sizes
    local_n = args.n
    X = X[(rank)*local_n:(rank+1)*local_n]
    N, d = X.shape
    assert(N == local_n)

    global_n = size * local_n

    #Convert X to the correct datatype
    X_data = np.asarray(X.data, dtype=np.float32)
    X_indices = np.asarray(X.indices, dtype=np.int32)
    X_indptr = np.asarray(X.indptr, dtype=np.int32)
    X = sp.csr_matrix( (X_data, X_indices, X_indptr), shape=(local_n, d))

    if rank == 0:
        print("Local Size:", X.shape, flush=True)

    timer = Profiler()
    record = Recorder()

    #Compute the true solution by exhaustive search
    tree = RKDT(data=X, levels=0, leafsize=2048, location="HOST")
    t_start = time.perf_counter()
    truth = tree.distributed_exact(Q, args.k)
    t_end = time.perf_counter()
    if rank == 0:
        print("Exact Search took: ", t_end-t_start, " (s)", flush=True)


    print("Truth:", truth)

    np.random.seed(args.seed)
    np.random.seed(150)

    forest = RKDForest(data=X, levels=args.levels, leafsize=args.leafsize, location=location)
    if args.overlap:
        approx = forest.overlap_search(args.k, ntrees=args.iter, ltrees = args.ltrees, truth=truth, cores=args.cores, blocksize=args.bs, blockleaf=args.bl, merge_flag=args.merge, threshold=0.95)
    else:
        approx = forest.all_search(args.k, ntrees=args.iter, ltrees = args.ltrees, truth=truth, cores=args.cores, blocksize=args.bs, blockleaf=args.bl, merge_flag=args.merge)

    if rank == 0:
        print(approx)
        timer.print()
        print("=======")
        record.print()
        
run()
