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

import argparse

parser = argparse.ArgumentParser(description="Test Sparse KNN")
parser.add_argument('-iter', type=int, default=10)
parser.add_argument('-dataset', default="url")
parser.add_argument('-bs', type=int, default=64)
parser.add_argument('-bl', type=int, default=128)
parser.add_argument('-cores', type=int, default=56)
parser.add_argument('-use_gpu', type=bool, default=0)
parser.add_argument('-levels', type=int, default=13)
parser.add_argument('-k', type=int, default=32)
parser.add_argument('-leafsize', type=int, default=512)
parser.add_argument('-ltrees', type=int, default=3)
parser.add_argument('-merge', type=int, default=1)
parser.add_argument('-overlap', type=int, default=1)
parser.add_argument('-seed', type=int, default=15)
parser.add_argument('-nq', type=int, default=10)

args = parser.parse_args()

mem = Memory("./mycache")

@mem.cache()
def get_url_data():
    t = time.time()
    data = load_svmlight_file(os.environ["SCRATCH"]+"/datasets/url/url_combined", n_features=3231961)
    t = time.time() - t
    print("It took ", t, " (s) to load the dataset")
    return data[0]

@mem.cache()
def get_avazu_data():
    t = time.time()
    data_app = load_svmlight_file(os.environ["SCRATCH"]+"/datasets/avazu/avazu-app", n_features=1000000)    
    print(data_app[0].shape)
    data_site = load_svmlight_file(os.environ["SCRATCH"]+"/datasets/avazu/avazu-site", n_features=1000000)
    print(data_site[0].shape)
    data = sp.vstack([data_app[0], data_site[0]])
    t = time.time() - t
    print("It took ", t, " (s) to load the dataset")
    return data

@mem.cache()
def get_kdd12_data():
    t = time.time()
    data = load_svmlight_file(os.environ["SCRATCH"]+"/datasets/kdd/kdd12", n_features=54686452)
    print(data[0].shape)
    t = time.time() - t
    print("It took ", t, " (s) to load the dataset")
    return data[0]


if args.dataset == "url":
    get_data = get_url_data 
elif args.dataset == "avazu":
    get_data = get_avazu_data
elif args.dataset == "kdd":
    get_data = get_kdd12_data

if args.use_gpu:
    location = "GPU"
else:
    location = "HOST"

def run():
    sparse = True
    k = args.k

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    print(rank, ", ", os.getpid(), flush=True)
    time.sleep(5)

    #Read data
    if rank == 0:
        print("Starting to Read Data", flush=True)

    t = time.time()
    X = get_data()
    t = time.time() - t 

    if rank == 0:
        print("Finished Reading Data: ", X.shape, flush=True)
        print("Reading data took: ", t," seconds", flush=True)

    Nmax = 2**27
    X = X[:Nmax]

    N, d = X.shape
    #Grab queries from start
    nq = 20
    Q = X[:nq]

    #Convert Q to the correct datatype
    q_data = np.asarray(Q.data, dtype=np.float32)
    q_indices = np.asarray(Q.indices, dtype=np.int32)
    q_indptr = np.asarray(Q.indptr, dtype=np.int32)
    Q = sp.csr_matrix( (q_data, q_indices, q_indptr), shape=(nq, d))

    #Grab local portion
    n_local = N//size

    start = (rank)*n_local
    end   = (rank+1)*n_local

    X = X[start:end]

    if rank == 0:
        print("Finished Preprocessing Data", flush=True)

    t = time.time()

    timer = Profiler()
    record = Recorder() 

    #Compute true solution with brute force on nq subset
    #C = X.copy()
    tree = RKDT(data=X, levels=0, leafsize=2048, location="HOST")
    truth = tree.distributed_exact(Q, k)
    t = time.time() - t

    if rank == 0:
        print("Exact Search took: ", t, " (s)", flush=True)

    #print("Truth:", truth)

    #timer.reset()
    #record.reset()

    #np.random.seed(100)
    np.random.seed(args.seed)
    forest = RKDForest(data=X, levels=args.levels, leafsize=args.leafsize, location=location)

    if args.overlap:
        approx = forest.overlap_search(k, ntrees=args.iter, ltrees = args.ltrees, truth=truth, cores=args.cores, blocksize=args.bs, blockleaf=args.bl, merge_flag=args.merge)
    else:
        approx = forest.all_search(k, ntrees=args.iter, ltrees = args.ltrees, truth=truth, cores=args.cores, blocksize=args.bs, blockleaf=args.bl, merge_flag=args.merge)

    if rank == 0:
        timer.print()
        print("=======")
        record.print()
        
run()








