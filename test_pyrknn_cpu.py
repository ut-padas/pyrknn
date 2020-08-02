from pyrknn.kdforest.mpi.tree import *
from pyrknn.kdforest.mpi.util import *
from pyrknn.kdforest.mpi.forest import *

from mpi4py import MPI

import numpy as np

import time
import platform

from sklearn.datasets import fetch_openml
from sklearn.preprocessing import normalize

import argparse

np.set_printoptions(precision=3)

def distance(R, Q):
    """Compute the distances between a reference set R (|R| x d) and query set Q (|Q| x d).

    Result:
        D -- |Q| x |R| resulting distance matrix.
        Note: Positivity not enforced for rounding errors in distances near zero.
    """
    D= -2* (Q @ R.T)                    #Compute -2*<q_i, r_j>, Note: Python is row-major

    Q2 = lib.linalg.norm(Q, axis=1)**2       #Compute ||q_i||^2
    R2 = lib.linalg.norm(R, axis=1)**2       #Compute ||r_j||^2

    D = D + Q2[:, np.newaxis]               #Add in ||q_i||^2 row-wise
    D = D + R2                              #Add in ||r_j||^2 colum-wise
    return D

def single_knn(gids, R, Q, k, lenv = None):
    #Note: My Pure python implementation is quite wasteful with memory.

    N, d, = Q.shape

    #Allocate storage space for neighbor ids and distances
    neighbor_list = lib.zeros([N, k])                #TODO: Change type to int
    neighbor_dist = lib.zeros([N, k])

    dist = distance(R, Q)                           #Compute all distances

    for q_idx in range(N):

        #TODO: Replace with kselect kernel
        #TODO: neighbor_dist, neighbor_list  = np.kselect(dist, k, key=gids)

        #Perform k selection on distance list
        neighbor_lids = lib.argpartition(dist[q_idx, ...], k)[:k]            #This performs a copy of dist and allocated lids
        neighbor_dist[q_idx, ...] = dist[q_idx, neighbor_lids]              #This performs a copy of dist
        neighbor_gids = gids[neighbor_lids]                                 #This performs a copy of gids

        #Sort and store the selected k neighbors
        shuffle_idx = lib.argsort(neighbor_dist[q_idx, ...])                 #This performs a copy of dist and allocates idx
        neighbor_dist[q_idx, ...] = neighbor_dist[q_idx, shuffle_idx]       #This performs a copy of dist
        neighbor_gids = neighbor_gids[shuffle_idx]                          #This performs a copy of gids

        neighbor_list[q_idx, ...] = neighbor_gids                           #This performs a copy of gids

    return neighbor_list, np.sqrt(neighbor_dist)

def neighbor_dist(a, b):
    a_list = a[0]
    a_dist = a[1]

    b_list = b[0]
    b_dist = b[1]

    Na, ka = a_list.shape
    Nb, kb = b_list.shape
    assert(Na == Nb)
    assert(ka == kb)

    changes = 0
    nn_dist = lib.zeros(Na)
    first_diff = lib.zeros(Na)
    knndist = 0
    for i in range(Na):
        changes += np.sum([1 if a_list[i, k] in b_list[i] else 0 for k in range(ka)])
        knndist = max(knndist, np.abs(a_dist[i, ka-1] - b_dist[i, kb-1])/a_dist[i, ka-1])

    perc = changes/(Na*ka)

    return perc, knndist

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

parser = argparse.ArgumentParser(description='Load file and compute Nearest Neighbors')
parser.add_argument('-file',
                    help='binary file to read')
parser.add_argument('-d', dest='dim', type=int,
                    help='The dimension of the dataset')
parser.add_argument('-k', dest='k', default=64, type=int, help='Number of nearest neighbors to find')
parser.add_argument('-exact', dest='exact', default=1000, type=int, help='Number of exact evaluations to verify')
 
args = parser.parse_args()
k = args.k

d = args.dim
# create a random matrix to index
data = np.fromfile(args.file, dtype=np.float32)
N = len(data)//d
data = np.reshape(data, (N, d))
#data = np.random.rand(N, d)
data = np.float32(data)

exactN = 1000

#Compute exact neighbors on a subset
A = np.copy(data)
tree = RKDForest(pointset=A, levels=1, leafsize=len(data), location="CPU", ntrees=1)
tree.build()
tree = tree.forestlist[0]
root = tree.nodelist[0]
Q = data[:exactN, ...]

exact_time = time.time()
truth = root.knn(Q, k)
exact_time = time.time() - exact_time

output = truth
print("TRUTH", output)

total_time = time.time()

tree = RKDForest(pointset=data, levels=20, leafsize=1024, comm=comm, location="GPU", ntrees=24, sparse=sparse)

approx = tree.aknn_all_build(k, cores=56, truth=output, until=False, ntrees=1)

total_time = time.time() - total_time

timer = Profiler()
timer.print()

print("Total Exact Time: ", exact_time)
print("Total Approx Time: ", total_time)
print(accuracy)

