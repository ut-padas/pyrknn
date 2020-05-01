from . import error as ErrorType
from . import util as Primitives
from .tree import *
import os


import numpy as np

if os.environ["PRKNN_USE_CUDA"] == '1':
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

    verbose = False #Note: This is a static variable

    def __init__(self, ntrees=1, levels=0, leafsize=None, pointset=None, location="CPU", comm=MPI.COMM_WORLD, sparse=False, N=None, d=None):
        """Initialize Randomized KD Forest.

        Keyword Arguments:
            ntrees -- number of trees to generate in the forest
            levels -- the maximum number of levels for all trees
            leafsize -- nodes are not split below size (2*leafsize +1)
            pointset -- the pointset for the nearest neighbor searches
        """
        if levels < 0:
            raise ErrorType.InitializationError('Invalid max levels parameter: Cannot build trees with '+str(levels)+' levels')

        if ntrees < 0:
            raise ErrorType.InitializationError('Invalid ntrees parameter: '+str(ntrees)+' ntrees')
        self.N = N
        self.d = d

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
        print("Running on device", self.device)

        if self.location == "CPU":
            self.lib = np
        elif self.location == "GPU":
            self.lib = cp
 
        if (pointset is not None):
            self.size = pointset.shape[0]
            self.gids = np.arange(self.size)

            if(self.sparse):
                local_data = np.asarray(pointset.data, dtype=np.float32)
                local_row = np.asarray(pointset.row, dtype=np.int32)
                local_col = np.asarray(pointset.col, dtype=np.int32)

                self.data = sp.coo_matrix( (local_data, (local_row, local_col) ), shape=(N, d))
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

                #Copy of location memory
                #local_data = self.lib.asarray([], dtype=np.float32)
                #local_indices = self.lib.asarray([], dtype=np.int32)
                #local_indptr = self.lib.asarray([], dtype=np.float32)
                
                #self.data = self.lib.csr_matrix( (local_data, local_indicies, local_indptr) )
            else:
                #Copy of data in CPU Memory
                self.data = np.asarray([], dtype=np.float32)
                #Copy of data in location memory
                #self.data = self.lib.asarray([], dtype=np.float32)



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
            tree = RKDT(pointset=self.data, levels=self.levels, leafsize=self.leafsize, location=self.location, comm=self.comm, sparse=self.sparse, N=self.N, d=self.d)
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

        #TODO: Key Area for PARLA Tasks

        for tree in self.forestlist:
            neighbors = tree.aknn(Q, k)

            if result is None:
                result = neighbors
            else:
                result = Primitives.merge_neighbors(result, neighbors, k)

            #TODO: Delay merge until after all searches (increase storage to O(|Q| x k ) x ntrees but increase potential task parallelism ?

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

        #TODO: Key Area for PARLA Tasks

        for tree in self.forestlist:
            neighbors = tree.aknn_all(k)

            if result is None:
                result = neighbors
            else:
                result = Primitives.merge_neighbors(result, neighbors, k)

            #TODO: Delay merge until after all searches (increase storage to O(|Q| x k ) x ntrees but increase potential task parallelism ?
        return result

    def aknn_all_build(self, k, verbose=False, blockleaf=10, blocksize=256, ntrees=1, cores=4, truth=None, until=False):

        Primitives.set_cores(cores)
        Primitives.reset_timing()

        v = (self.verbose or verbose)
        result = None
        rank = self.comm.Get_rank()
        #TODO: Key Area for PARLA Tasks

        test = None
        if truth:
            nq = truth[0].shape[0]
            prev = np.empty([nq, k], dtype=np.int32)
        else:
            prev = None

        if until:
            self.ntrees = 100

        print("...entering loop", flush=True)
        print("starting search...", flush=True)

        total_t = time.time()
        for it in range(self.ntrees):
            print(rank, "Begin tree", flush=True)
            #X = np.copy(self.data)
            X = self.data
            tree = RKDT(pointset=X, levels=self.levels, leafsize=self.leafsize, location=self.location, comm=self.comm, sparse=self.sparse, N=self.N, d=self.d)

            if self.location == "CPU":
                build_t = time.time()
                tree.build()
                build_t = time.time() - build_t
                print(rank, "Build_t CPU Tree", build_t, flush=True)
 
                search_t = time.time()
                neighbors = tree.aknn_all(k)
                search_t = time.time() - search_t
                print(rank, "Search_t CPU", search_t, flush=True)

                Primitives.aknn_t += (search_t) + (build_t)

            else:
                dist_t = time.time()
                #tree.dist_build()
                #tree.dist_build_lil()

                if self.sparse:
                    tree.dist_build_sparse()
                else:
                    tree.dist_build()

                dist_t = time.time() - dist_t
                print(rank, "Build_Dist_t", dist_t, flush=True)
                Primitives.dist_build_t += dist_t

                aknn_t = time.time()
                if self.sparse:
                    print(rank, "type", type(tree.data))
                    print(rank, "data", (tree.data.data))
                    print(rank, "ptr", (tree.data.indptr))
                    print(rank, "idx", (tree.data.indices))
                    print(rank, "matrix", (tree.data.toarray()))
                    print(rank, "gids", tree.host_real_gids)

                    neighbors = Primitives.sparse_knn(tree.host_real_gids, tree.data, tree.levels-tree.dist_levels, ntrees, k, blockleaf, blocksize, self.device)
                else:
                    neighbors = Primitives.dense_knn(tree.host_real_gids, tree.data, tree.levels - tree.dist_levels, ntrees, k, blocksize, self.device) 
                #[0, 1, 2, 3, 4, 5]
                #[2, 3, 5, 10, 3, 6]               
 
                aknn_t = time.time() - aknn_t
                print("aknn_t", aknn_t)
                Primitives.aknn_t += aknn_t

                #print("gids", tree.host_gids)
                #print("rids", tree.host_real_gids)

                #neighbor_id = neighbors[0]
                #neighbor_dist = neighbors[1]

                #neighbor_id = tree.host_real_gids[neighbors[0][:]]
                #neighbor_dist = neighbors[1][:]

                #neighbors = (neighbor_id, neighbor_dist)

            #print(rank, "neighbors", neighbors, flush=True)

            #print(rank, "host gids", tree.host_gids, flush=True)
            #print(rank, "host gids", tree.host_real_gids, flush=True)

            if self.location == "GPU":
                tree.real_gids = tree.host_real_gids
            #    real_gids = tree.host_real_gids[tree.host_gids]
            #    neighbor_list = real_gids[neighbors[0]]
            #    print("gids", tree.host_real_gids)
            #    print("doblb", neighbor_list, flush=True)
            #    neighbors = (neighbor_list, neighbors[1])
                #neighbor_list2 = tree.host_real_gids[neighbors[0]]
                #print("sng", neighbor_list2, flush=True)
            #print(rank, "gids", tree.real_gids)
            #if 0 in tree.real_gids:
            #    print("0 is on rank", rank)
            #    loc = np.where(tree.real_gids==0)
            #    print("at loc", loc[0])
            #    loc_2 = np.where(tree.gids==loc[0])
            #    print("The datapoint is", tree.data[loc_2])
            
            #DEBUG: Merge with self to sort
            #neighbors = Primitives.merge_neighbors(neighbors, neighbors, k)
            
            #print("Before redist", rank, neighbors, flush=True)
            #if 0 in tree.real_gids:
            #    print("0 is on rank", rank)
            #    loc = np.where(tree.real_gids==0)
            #    print("zero", neighbors[0][loc, ...])
            #    print("zero", neighbors[1][loc, ...])

            dist_t = time.time()
            gids, neighbors = tree.redist(neighbors)
            dist_t = time.time() - dist_t
            #print(rank, "redistribute_t", dist_t, flush=True)

            Primitives.redistribute_t += dist_t

            #print("New GIDS", rank, gids, flush=True)

            #print("Results after Redist:", rank, neighbors, flush=True)
            """
            copy_t = time.time()
            if self.location == "GPU":
                neighbor_ids = self.lib.array(neighbors[0])
                neighbor_dist = self.lib.array(neighbors[1])

                neighbors_device = (neighbor_ids, neighbor_dist)
                del neighbors
                neighbors = neighbors_device

                if result:
                    res_ids = self.lib.array(result[0])
                    res_dist = self.lib.array(result[1])

                    res_device = (res_ids, res_dist)
                    del result
                    result = res_device

            copy_t = time.time() - copy_t
            print(rank, "copy_to_gpu_t", copy_t, flush=True)
            Primitives.copy_gpu_t += copy_t
            """

            if result is None:
                result = neighbors
            else:
                #print("result", result)
                #print("neighbors", neighbors)
                merge_t = time.time()
                result = Primitives.merge_neighbors(result, neighbors, k)
                merge_t = time.time() - merge_t
                #print(rank, "merge_t", merge_t, flush=True)
                Primitives.merge_t += merge_t
                #print("after merge", result)
            """
            #Copy results back to host to save room on gpu
            if self.location == "GPU":
                res_ids = self.lib.asnumpy(result[0])
                res_dist = self.lib.asnumpy(result[1])
                res_host = (res_ids, res_dist)
                del result

                result = res_host
            """

            memory_t = time.time()
            del tree
            del neighbors

            #mempool = cp.get_default_memory_pool()
            #mempool.free_all_blocks()
            #gc.collect()
            memory_t = time.time() - memory_t
            print("memory_t", memory_t)

            gap = 5
            check_t = time.time()
            if truth is not None:
                flag = False
                tlist, tdist = truth
                nq = tlist.shape[0]
                rlist, rdist = result

                test = (rlist[:nq, ...], rdist[:nq, ...])

                if it>gap:
                    print("test", test)
                    print("temp", prev)

                if prev is not None:
                    diff = nq*k - np.sum(test[0] == prev)
                    print("Difference", diff)
                else:
                    diff = nq*k

                Primitives.diff.append(diff)

                prev = np.copy(test[0])
    
                acc = Primitives.neighbor_dist(truth, test)
                Primitives.accuracy.append(acc)
                print("Iteration:", it, "Checking acc:", acc)

                if it > gap:
                    diff = abs(diff - Primitives.diff[it-gap]) < 1 and diff < 0.05*nq*k
                else:
                    diff = False

                if until:
                    if acc[0] > 0.95 or diff:
                        print(acc, diff)
                        flag = True
            else:
                flag = None

            if until:
                flag = self.comm.bcast(flag, root=0)
            check_t = time.time() - check_t 
            Primitives.check_t += check_t

            print("check_t", check_t)

            if flag:
                break
                

        total_t = time.time() - total_t
        print("ALLKNN TIME", total_t)

        Primitives.total_t = total_t

        Primitives.timing()

        return result


    def grow(l):
        """Add l more trees to the forest"""

        #TODO: Key Area for PARLA Tasks
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


