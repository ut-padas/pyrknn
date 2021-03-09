from . import error as ErrorType
from . import util as Primitives
from .tree import *
import os


import numpy as np

if os.environ["PYRKNN_USE_CUDA"] == '1':
    import cupy as cp
else:
    import numpy as cp

from collections import defaultdict
import gc
import time
import scipy.sparse as sp

from mpi4py import MPI

class RKDForest:
    """Class for a collection of RKD Trees for Nearest Neighbor searches"""

    verbose = False

    def __init__(self, ntrees=1, levels=0, leafsize=None, pointset=None, location="CPU", comm=MPI.COMM_WORLD, sparse=False, N=None, d=None):
        if levels < 0:
            raise ErrorType.InitializationError('Invalid max levels parameter: Cannot build trees with '+str(levels)+' levels')
        if ntrees < 0:
            raise ErrorType.InitializationError('Invalid ntrees parameter: '+str(ntrees)+' ntrees')

        self.comm = comm
        self.id = id(self)
        self.levels = levels
        self.leafsize = leafsize
        self.ntrees = ntrees
        self.location = location
        self.sparse = sparse
        rank = self.comm.Get_rank()

        self.device = rank%4
        Primitives.set_device(self.device);

        if self.location == "GPU":
            self.lib = cp
        else:
            self.lib = np


        if N is not None:
            self.N = N
        else:
            self.N = None

        if d is not None:
            self.d = d
        else:
            self.d = None

        if (pointset is not None):
            self.size = pointset.shape[0]
            self.gids = np.arange(self.size)

            if(self.sparse):
                local_data = np.asarray(pointset.data, dtype=np.float32)
                local_row = np.asarray(pointset.row, dtype=np.int32)
                local_col = np.asarray(pointset.col, dtype=np.int32)

                self.data = sp.coo_matrix( (local_data, (local_row, local_col) ), shape=(N, d))
                print("Out of the Forest", self.data.shape)
            else:
                self.data = np.asarray(pointset, dtype=np.float32)

            if (leafsize is None):
                self.leafsize = self.size
            if (self.ntrees == 0):
                self.empty = True
            else:
                self.empty = False

            if self.leafsize < 0:
                raise ErrorType.InitializationError('Invalid leafsize parameter: '+str(self.leafsize))
        else:
            self.empty= True
            self.ntrees = 0
            if(self.sparse):
                #Copy of CPU Memory
                local_data = np.asarray([], dtype=np.float32)
                local_row = np.asarray([], dtype=np.int32)
                local_col = np.asarray([], dtype=np.float32)

                self.data = sp.coo_matrix( (local_data, (local_row, local_col) ))
            else:
                self.data = np.asarray([], dtype=np.float32)

        self.built=False


    def build(self):
        """Build the RKD Forest"""

        #checking parameters to ensure valid initialization
        if self.built:
            raise ErrorType.InitializationError('You cannot build a forest that has already been built. Please clear() or use grow()')
        if self.empty:
            raise ErrorType.InitializationError('Cannot build an empty forest')
        if self.ntrees < 0:
            raise ErrorType.InitializationError('Invalid size parameter: Cannot build a forest of size '+str(self.size))
        if self.leafsize <= 0:
            raise ErrorType.InitializationError('Invalid leaf size parameter: Cannot build trees with leaf size '+str(self.leafsize))
        if self.levels < 0:
            raise ErrorType.InitializationError('Invalid max levels parameter: Cannot build trees with '+str(self.levels)+' levels')

        #TODO: Key area for PARLA Tasks

        self.forestlist = [None]*self.ntrees

        for i in range(self.ntrees):
            X = self.data
            tree = RKDT(pointset=X, levels=self.levels, leafsize=self.leafsize, location=self.location, comm=self.comm, sparse=self.sparse, N=self.N, d=self.d)
            tree.build()
            self.forestlist[i] = tree

        self.built=True

    def knn(self, Q, k):
        """Perform an exact exhaustive knn search using the root node of the first tree.

        Arguments:
            Q -- N x d query matrix
            k -- number of nearest neighbors (req. k < leafsize)

        Return:
            (neighbor_list , neigbor_dist)
            neighbor_list -- |Q| x k list of neighbor gids. In local ordering og the query points.
            neighbor_dist -- the corresponding distances
        """

        first = self.forestlist[0]
        return first.knn(Q, k)

    def dist_exact(self, Q, k):
        first = self.forestlist[0]
        return first.dist_exact(Q, k)


    def aknn(self, Q, k, verbose=False):
        """Perform an approximate knn search over all trees, merging results.

        Arguments:
            Q -- N x d query matrix
            k -- number of nearest neighbors (req. k < leafsize)

        Return:
            (neighbor_list , neigbor_dist)
            neighbor_list -- |Q| x k list of neighbor gids. In local ordering og the query points.
            neighbor_dist -- the corresponding distances
        """
        v = (self.verbose or verbose)
        result = None

        for tree in self.forestlist:
            neighbors = tree.aknn(Q, k)

            if result is None:
                result = neighbors
            else:
                result = Primitives.merge_neighbors(result, neighbors, k)

        return result

    def aknn_all(self, k, verbose=False):
        """Perform an approximate all knn search over all trees in forest, merging results.

        Arguments:
            Q -- N x d query matrix
            k -- number of nearest neighbors (req. k < leafsize)

        Return:
            (neighbor_list , neigbor_dist)
            neighbor_list -- |Q| x k list of neighbor gids. In local ordering og the query points.
            neighbor_dist -- the corresponding distances
        """
        v = (self.verbose or verbose)
        result = None

        for tree in self.forestlist:
            neighbors = tree.aknn_all(k)

            if result is None:
                result = neighbors
            else:
                result = Primitives.merge_neighbors(result, neighbors, k)

        return result

    def search(self, k, verbose=False, blockleaf=10, blocksize=256, ntrees=1, cores=4, truth=None, until=False, until_max=100, nq=1000, gap=5, threshold=0.95):
        timer = Primitives.Profiler()
        record = Primitives.Recorder()

        timer.push("Forest: AKNN All")

        timer.push("Forest: Setup")
        Primitives.set_cores(cores)

        v = (self.verbose or verbose)

        result = None

        rank = self.comm.Get_rank()
        size = self.comm.Get_size()
        sparse = self.sparse
        N = self.data.shape[0]

        if truth:
            nq = truth[0].shape[0]
        else:
            nq = nq if nq < N else np.log(N)

        prev = ( np.empty([nq, k], dtype=np.int32 ), np.empty( [nq, k], dtype=np.float32 ) )

        converge_log = np.ones(gap, dtype=np.int32)

        #Set the maximum number of iterations
        if until:
            self.ntrees = until_max

        timer.pop("Forest: Setup")

        #Iterate over the tree set.
        for it in range(self.ntrees):

            #Build Tree

            timer.push("Forest: Build Tree")
            if not sparse:
                X = np.copy(self.data)
            else:
                X = self.data.copy()
            tree = RKDT(pointset=X, levels=self.levels, leafsize=self.leafsize, location=self.location, comm=self.comm, sparse=self.sparse, N=self.N, d=self.d)
            tree.build()
            timer.pop("Forest: Build Tree")

            print("Searching Neighbors")
            #Compute Neighbors
            timer.push("Forest: Compute Neighbors")
            if (self.location == "CPU" or self.location == "PYTHON" ) and (not sparse):
                neighbors = tree.aknn_all(k)
            elif self.location == "CPU" and sparse:
                print("Starting Sparse Search")
                neighbors = Primitives.cpu_sparse_knn(tree.host_real_gids, tree.data, tree.levels-tree.dist_levels, ntrees, k, blocksize)
            elif self.location =="GPU" and self.sparse:
                neighbors = Primitives.gpu_sparse_knn(tree.host_real_gids, tree.data, tree.levels-tree.dist_levels, ntrees, k, blockleaf, blocksize, self.device)
            elif self.location == "GPU" and (not self.sparse):
                neighbors = Primitives.dense_knn(tree.host_real_gids, tree.data, tree.levels - tree.dist_levels, ntrees, k, blocksize, self.device)
            else:
                raise Exception("Your compute location is not valid. Select GPU or CPU Runtime")
            timer.pop("Forest: Compute Neighbors")

            timer.push("Forest: Redistribute")
            #Redistribute
            if size > 1:
                gids, neighbors = tree.redist(neighbors)
            timer.pop("Forest: Redistribute")

            del X
            del tree

            #Merge
            timer.push("Forest: Merge")
            if result is None:
                result = Primitives.merge_neighbors(neighbors, neighbors, k)
            else:
                result = Primitives.merge_neighbors(result, neighbors, k)
            timer.pop("Forest: Merge")

            #print("Result", result)
            #print("Truth", truth)

            break_flag = False

            #Compare against true NN
            timer.push("Forest: Compare")
            if ( truth is not None ) and ( rank == 0 ) :
                rlist, rdist = result
                test = (rlist[:nq, ...], rdist[:nq, ...])

                acc = Primitives.neighbor_dist(truth, test)
                Primitives.accuracy.append( acc )
                record.push("Recall", acc[0])
                record.push("Distance", acc[1])

                print("Iteration:", it, "Recall:", acc)

                if until and acc[0] > threshold:
                    break_flag = True

            if ( truth is None ) and until and ( rank == 0) :

                idx = it % gap

                rlist, rdist = result
                test = (rlist[:nq, ...], rdist[:nq, ...])

                if prev is not None:
                    diff = ( nq*k - np.sum(test[0] == prev[0]) ) / (nq*k)
                else:
                    diff = 1

                print("Iteration:", it, "Change %:", diff)

                Primitives.diff.append(diff)

                prev = ( np.copy(test[0]), np.copy(test[1]) )

                converge_log[idx] = 1 if diff < 0.05 else 0

                if np.sum(converge_log) > gap - 1:
                    break_flag = True

            break_flag = self.comm.bcast(break_flag, root=0)
            
            if break_flag:
                break

            timer.pop("Forest: Compare")

        timer.pop("Forest: AKNN All")
        return result


    def grow(l):
        """Add l more trees to the forest"""

        for i in range(l):
            tree = RKDT(pointset=self.data, levels=self.levels, leafsize=self.leafsize, location=self.location, comm=self.comm)
            self.treelist.append(tree)
        self.ntrees = len(self.treelist)

    def clear():
        """Clear the forest."""

        if self.built:
            self.forestlist = []
        self.data = np.asarray([])
        self.ntrees = 0
        self.size = 0
        self.empty = True

    def check_accuracy(self, Q=None, k=None):
        """Perform a comparison of knn accuracy between the approximate and exact search.

        Keywork Arguments:
            Q -- N x d query matrix
            k -- number of nearest neighbors (req. k < leafsize)

        if k not set, uses leafsize - 1
        if Q not set, performs all nearest neighbor search

        """
        if k is None:
            k = self.leafsize-1

        if Q is None:
            #Perform All Nearest Neighbor Search and Compare
            result = self.all_nearest_neighbor(k)

            first = self.forestlist[0]
            truth = first.exact_all_nearest_neighbors(k)
        else:
            #Perform query with the N, d dimensional query points in Q (N x d)
            result = self.aknn(Q, k)
            truth = self.knn(Q, k)

        return Primitives.neighbor_dist(truth, result)


    def clear():
        if self.built:
            self.forestlist = []
        self.data = np.asarray([])
        self.ntrees = 0
        self.size = 0
        self.empty = True


