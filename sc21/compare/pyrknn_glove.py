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
k = 32
t = 0
nq = 1000
import h5py

path = "/scratch1/06081/wlruys/datasets/glove/glove-50-angular.hdf5"
f = h5py.File(path, 'r')
X = np.array(f["train"])

Q = X[:nq]

N, d = X.shape

timer = Profiler()
record = Recorder()

#Compute true solution with brute force on nq subset
#C = X.copy()
tree = RKDT(data=X, levels=0, leafsize=2048, location="HOST")
truth = tree.distributed_exact(Q, k)

#print("Truth:", truth)

timer.reset()
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

