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
from scipy.sparse import vstack as sparse_stack

import argparse

parser = argparse.ArgumentParser(description="Test Sparse KNN")
parser.add_argument('-n', type=int, default=2**22)
parser.add_argument('-d', type=int, default=15)
parser.add_argument('-iter', type=int, default=120)
parser.add_argument('-dataset', default="gauss")
parser.add_argument('-bs', type=int, default=64)
parser.add_argument('-bl', type=int, default=128)
parser.add_argument('-cores', type=int, default=56)
parser.add_argument('-use_gpu', type=bool, default=0)
parser.add_argument('-levels', type=int, default=13)
parser.add_argument('-k', type=int, default=32)
parser.add_argument('-leafsize', type=int, default=1024)
parser.add_argument('-ltrees', type=int, default=1)
parser.add_argument('-q', type=int, default=100)
parser.add_argument('-merge', type=int, default=1)
parser.add_argument('-overlap', type=int, default=0)
parser.add_argument('-seed', type=int, default=15)
parser.add_argument('-nq', type=int, default=1000)
args = parser.parse_args()

if args.use_gpu:
    location = "GPU"
else:
    location = "HOST"

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
k = 64
t = 0
nq = 100

def read_truth(name, k):

    id_file = name+"_nborID_100.bin.npy"
    dist_file =name+"_nborDist_100.bin.npy"

    #truthID = np.fromfile(id_file, dtype=np.int32)
    #truthDist = np.fromfile(dist_file, dtype=np.float32)
    truthID = np.load(id_file)
    truthDist = np.load(dist_file)

    #truthID = truthID.reshape((len(truthID)//k, k))
    #truthDist = truthID.reshape(truthID.shape)
    print("Truth Shape: ", truthID.shape)

    truth = (truthID, truthDist)
    return truth

print("Starting Script", flush=True)
mem = Memory("./mycache")
name = os.environ["SCRATCH"]+"/comparison/avazu/"
name = "avazu"

@mem.cache()
def get_data():
    t = time.time()
    data_app = load_svmlight_file(os.environ["SCRATCH"]+"/datasets/avazu/avazu-app", n_features=1000000)
    data_site = load_svmlight_file(os.environ["SCRATCH"]+"/datasets/avazu/avazu-site", n_features=1000000)
    print(data_app[0], data_app[1])
    print(data_app[0].shape, data_app[1].shape)
    t = time.time() - t
    print("It took ", t, " (s) to load the dataset")
    return data_app[0], data_site[0]

print("Starting to Read Data", flush=True)
X_app, X_site = get_data()
print(X_app.shape, X_site.shape)
X = sparse_stack([X_app, X_site])

N, d = X.shape
print(X)

print("Finished Reading Data", flush=True)
k = 64
N  = X.shape[0]
d  = X.shape[1]
print("Data shape: ", (N, d))
truth = read_truth(name, k)

local_size = N//size
X = X[rank*local_size:(rank+1)*local_size]

timer = Profiler()
record = Recorder() 

#Compute true solution with brute force on nq subset
#C = X.copy()
#tree = RKDT(data=X, levels=0, leafsize=2048, location="HOST")
#truth = tree.distributed_exact(Q, k)

#print("Truth:", truth)

#timer.reset()
#record.reset()

np.random.seed(args.seed)
#np.random.seed(15)
forest = RKDForest(data=X, levels=args.levels, leafsize=args.leafsize, location=location)
if args.overlap:
    approx = forest.overlap_search(k, ntrees=args.iter, ltrees = args.ltrees, truth=truth, cores=args.cores, blocksize=args.bs, blockleaf=args.bl, merge_flag=args.merge)
else:
    approx = forest.all_search(k, ntrees=args.iter, ltrees = args.ltrees, truth=truth, cores=args.cores, blocksize=args.bs, blockleaf=args.bl, merge_flag=args.merge)

if rank == 0:
    timer.print()
    record.print()

