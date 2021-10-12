from . import error as ErrorType
from . import util as Primitives
from .tree import *
import os

import numpy as np

if Primitives.use_cuda:
    import cupy as cp
else:
    import numpy as cp

from collections import defaultdict
import gc
import time
import scipy.sparse as sp

from mpi4py import MPI
import pickle
import concurrent.futures
import psutil

def copy(data, sparse_flag):
    if sparse_flag:
        return data.copy()
    else:
        return np.copy(data)

def distributed_tree_task(tree_args):
    t = time.time()
    X, levels, leafsize, rank = tree_args

    #Build Distributed Tree (Assign IDs)
    tree = RKDT(data=X, levels=levels, leafsize=leafsize)
    tree.distributed_build()

    #Redistribute Coordinate Data
    tree.collect_data()
    return tree

def search_task_dense(search_args):
    X, gids, k, local_levels, blocksize, device, rank, ltrees = search_args

    #Perform GPU Dense Search
    neighbors = Primitives.gpu_dense_knn(gids, X, local_levels, ltrees, k, blocksize,device)
    return neighbors

def search_task_sparse(search_args):
    X, gids, k, local_levels, blocksize, blockleaf, device, rank, ltrees = search_args

    #Perform GPU Sparse Search
    neighbors = Primitives.gpu_sparse_knn(gids, X, local_levels, ltrees, k, blockleaf, blocksize,device)
    return neighbors

class RKDForest:
    """Class for a collection of RKDTrees for  Nearest Neighbor searches"""

    verbose = False

    def __init__(self, ntrees=1, levels=0, leafsize=None, data=None, location="HOST", comm=MPI.COMM_WORLD, cores=None):

        #Set number of CPU cores to use
        if cores is None:
            self.cores = psutil.cpu_count(logical=False)
        else:
            self.cores = cores

        #Setup MPI Communicator
        if comm is None:
            self.comm = MPI.COMM_WORLD
        else:
            self.comm = comm

        mpi_size = self.comm.Get_size()
        rank = self.comm.Get_rank()
        self.mpi_size = mpi_size
        self.rank = rank

        #Assign local precision
        self.lprec = np.int32

        #Assign each forest object a unique id
        self.loc_id = id(self)

        #Shared same id as on rank 0
        self.glb_id = 0
        if self.rank == 0:
            self.glb_id = id(self)
        self.glb_id = self.comm.bcast(self.glb_id)

        #Get the size of locally owned points
        self.local_size = data.shape[0]
        self.local_ids = np.arange(self.local_size)

        #Setup tree parameters
        self.levels = levels
        self.leafsize = leafsize
        self.ntrees = ntrees
        self.location = location

        sparse_flag = isinstance(data, sp.csr.csr_matrix)
        dense_flag = isinstance(data, np.ndarray)
        assert(sparse_flag or dense_flag)

        self.sparse_flag = sparse_flag
        self.gpu_flag = (self.location == "GPU")

        #Set device id for search kernels

        rank = self.comm.Get_rank()

        self.device = 0
        if( self.gpu_flag and Primitives.use_cuda):
            ndevices = cp.cuda.runtime.getDeviceCount()
            self.device = rank % ndevices

        Primitives.set_device(self.device)

        #Setup data
        self.host_data = None
        if self.sparse_flag:
            N, d = data.shape
            local_value = np.asarray(data.data, dtype=np.float32)
            local_rowptr = np.asarray(data.indptr, dtype=self.lprec)
            local_colidx = np.asarray(data.indices, dtype=self.lprec)

            self.host_data = sp.csr_matrix(
                (local_value, local_colidx, local_rowptr), shape=(N, d))
        else:
            self.host_data = np.asarray(data, dtype=np.float32)

        self.dim = self.host_data.shape[1]

    def _build_and_search(self, k, current_tree = None, cores=8, blocksize=64, blockleaf=128, ltrees=3, verbose=False, overlap=False):
        timer = Primitives.Profiler()

        mpi_size = self.comm.Get_size()
        rank = self.comm.Get_rank()

        #Initialize output
        neighbors = None
        next_tree = None

        if current_tree is None:
            timer.push("Forest: Build Dist Tree")
            X = copy(self.host_data, self.sparse_flag)
            tree = RKDT(data=X, levels=self.levels, leafsize=self.leafsize)

            if mpi_size > 1:
                tree.distributed_build()
            timer.pop("Forest: Build Dist Tree")

            timer.push("Forest: Distribute Coordinates")
            if mpi_size > 1:
                tree.collect_data()
            timer.pop("Forest: Distribute Coordinates")
            current_tree = tree

        timer.push("Forest: Evaluate")

        if overlap and self.gpu_flag:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                tree_args = (self.host_data, self.levels, self.leafsize, rank)
                tree_future = executor.submit(distributed_tree_task, tree_args)

                if self.sparse_flag:
                    search_args = (current_tree.host_data, current_tree.global_ids, k, current_tree.local_levels, blocksize, blockleaf, self.device, rank, ltrees)
                    search_future = executor.submit(search_task_sparse, search_args)
                else:
                    search_args = (current_tree.host_data, current_tree.global_ids, k, current_tree.local_levels, blocksize, self.device, rank, ltrees)
                    search_future = executor.submit(search_task_dense, search_args)

                next_tree = tree_future.result()
                neighbors = search_future.result()
        else:
            #TODO: Either support int64 or change to local_ids and update in python
            if self.gpu_flag and (not self.sparse_flag):
                neighbors = Primitives.gpu_dense_knn(tree.global_ids, tree.host_data, tree.local_levels, ltrees, k, blocksize, self.device)
            elif self.gpu_flag and self.sparse_flag:
                neighbors = Primitives.gpu_sparse_knn(tree.global_ids, tree.host_data, tree.local_levels, ltrees, k, blockleaf, blocksize, self.device)
            elif (not self.gpu_flag) and (not self.sparse_flag):
                tree.build_local()
                neighbors = tree.search_local(k)
            elif (not self.gpu_flag) and (self.sparse_flag):
                n, d = tree.host_data.shape
                neighbors = Primitives.cpu_sparse_knn_3(tree.global_ids, tree.host_data.indptr, tree.host_data.indices, tree.host_data.data, len(tree.host_data.data), tree.local_levels, ltrees, k, blocksize, n, d, cores)

        timer.pop("Forest: Evaluate")

        return neighbors, current_tree, next_tree


    def search(self, k, ntrees=5, truth=None, cores=8, blocksize=64, blockleaf=128, ltrees=3, threshold=0.95, merge_flag=True, verbose=False, evaluate_truth=False, evaluate_diff=False, sample=None, k_acc_list=None):
        timer = Primitives.Profiler()
        record = Primitives.Recorder()

        timer.push("Forest: Search")

        result = None

        if merge_flag:
            merge_location = self.location
        else:
            merge_location = "HOST"

        rank = self.comm.Get_rank()
        mpi_size = self.comm.Get_size()

        if truth is not None:
            nq = truth[0].shape[0]
            assert(nq == truth[1].shape[0])
            assert(truth[0].shape[1] == truth[1].shape[1])
            assert(truth[0].shape[1]>=k)
            evalute_truth = True

        if evalute_truth and (sample > 0):
            local_size = self.host_data.shape[0]
            d = self.host_data.shape[1]

            active_subset_idx = np.random.choice(local_size, size=sample, replace=False)
            local_Q = self.host_data[active_subset_idx, :]

            truth = distributed_exact(local_Q, k)

        current_tree = None
        next_tree = None

        for it in range(ntrees):

            #Build Distributed Tree, Search, and if overlaping build next distributed tree
            neighbors, current_tree, next_tree = _build_and_search_local(self, k, current_tree, cores, blocksize, blockleaf, ltrees, verbose, overlap)

            #Sort to check intermediary local neigbors
            #neighbors = Primitives.merge_neighbors(neighbors, neighbors, k)
            #print(rank, "Neighbors before merge: : ", neighbors, flush=True)

            #Redistribute
            timer.push("Forest: Redistribute")
            gids, neighbors = current_tree.redistribute_results(neighbors)
            timer.pop("Forest: Redistribute")

            #Merge
            timer.push("Forest: Merge")
            if result is None:
                result = Primitives.merge_neighbors(neighbors, neighbors, k, merge_location, cores)
            else:
                result = Primitives.merge_neighbors(result, neighbors, k, merge_location, cores)
            timer.pop("Forest: Merge")

            #Update active tree
            current_tree = next_tree

            #Run Comparison on Test Points
            timer.push("Forest: Compare")
            if ( truth is not None ) and ( rank == 0 ) :
                rlist, rdist = result
                test = (rlist[:nq, ...], rdist[:nq, ...])

                if k_acc_list is not None:
                    if k_acc_list == "Default":
                        acc = Primitives.accuracy_stride(truth, test, None)
                    else:
                        acc = Primitives.accuracy_stride(truth, test, k_acc_list)
                else:
                    acc = Primitives.check_accuracy(truth, test)

                record.push("Recall", acc[0])
                record.push("Distance", acc[1])
                record.push("Similarity", acc[2])

                if verbose:
                    print("Iteration:", it, "Recall:", acc, flush=True)


            timer.pop("Forest: Compare")
        timer.pop("Forest: Search")

        return result





    def distributed_exact(self, Q, k):
        tree = RKDT(data=self.host_data)
        truth = tree.distributed_exact(Q, k)
        return truth







    #These 'overlap_search' and 'all_search' functions will be deprecated in an upcoming release
    def overlap_search(self, k, ntrees=5, truth=None, cores=8, blocksize=64, blockleaf=128, ltrees=3, threshold=0.95, merge_flag=True, verbose=False):
        timer = Primitives.Profiler()
        record = Primitives.Recorder()

        timer.push("Total Time")
        result = None

        if merge_flag:
            merge_location = self.location
        else:
            merge_location = "HOST"

        rank = self.comm.Get_rank()
        mpi_size = self.comm.Get_size()

        if truth is not None:
            nq = truth[0].shape[0]
            assert(nq == truth[1].shape[0])
            assert(truth[0].shape[1] == truth[1].shape[1])
            assert(truth[0].shape[1]>=k)

        current_tree = None

        for it in range(ntrees):

            if current_tree is None:
                timer.push("Build Dist Tree")
                X = copy(self.host_data, self.sparse_flag)
                tree = RKDT(data=X, levels=self.levels, leafsize=self.leafsize)
                tree.distributed_build()
                timer.pop("Build Dist Tree")

                timer.push("Distribute Coordinates")
                tree.collect_data()
                timer.pop("Distribute Coordinates")
                current_tree = tree

            timer.push("Evaluate")

            with concurrent.futures.ThreadPoolExecutor() as executor:
                tree_args = (self.host_data, self.levels, self.leafsize, rank)
                tree_future = executor.submit(distributed_tree_task, tree_args)

                if self.sparse_flag:
                    search_args = (current_tree.host_data, current_tree.global_ids, k, current_tree.local_levels, blocksize, blockleaf, self.device, rank, ltrees)
                    search_future = executor.submit(search_task_sparse, search_args)
                else:
                    search_args = (current_tree.host_data, current_tree.global_ids, k, current_tree.local_levels, blocksize, self.device, rank, ltrees)
                    search_future = executor.submit(search_task_dense, search_args)

                next_tree = tree_future.result()
                neighbors = search_future.result()

            timer.pop("Evaluate")

            #Sort to check
            #neighbors = Primitives.merge_neighbors(neighbors, neighbors, k)
            #print(rank, "Neighbors before merge: : ", neighbors, flush=True)

            #Redistribute
            timer.push("Forest: Redistribute")
            gids, neighbors = current_tree.redistribute_results(neighbors)
            timer.pop("Forest: Redistribute")

            #Merge
            timer.push("Forest: Merge")
            if result is None:
                result = Primitives.merge_neighbors(neighbors, neighbors, k, merge_location, cores)
            else:
                result = Primitives.merge_neighbors(result, neighbors, k, merge_location, cores)
            timer.pop("Forest: Merge")

            current_tree = next_tree

            timer.push("Forest: Compare")
            if ( truth is not None ) and ( rank == 0 ) :
                rlist, rdist = result
                test = (rlist[:nq, ...], rdist[:nq, ...])

                acc = Primitives.check_accuracy(truth, test)
                Primitives.accuracy.append( acc )
                record.push("Recall", acc[0])
                record.push("Distance", acc[1])

                if verbose:
                    print("Iteration:", it, "Recall:", acc, flush=True)
            timer.pop("Forest: Compare")
        timer.pop("Total Time")

        return result

    def all_search(self, k, ntrees=5, truth=None, cores=56, blocksize=64, blockleaf=128, ltrees=3, threshold=0.95, merge_flag=True, verbose=True):
        timer = Primitives.Profiler()
        record = Primitives.Recorder()

        timer.push("Forest: Search")
        result = None

        if merge_flag:
            merge_location = self.location
        else:
            merge_location = "HOST"

        rank = self.comm.Get_rank()
        mpi_size = self.comm.Get_size()
        break_flag = False

        if truth is not None:
            nq = truth[0].shape[0]
            assert(nq == truth[1].shape[0])
            assert(truth[0].shape[1] == truth[1].shape[1])
            assert(truth[0].shape[1]>=k)

        for it in range(ntrees):

            #Build Tree
            timer.push("Forest: Build Dist Tree")
            if mpi_size > 1:
                X = copy(self.host_data, self.sparse_flag)
                tree = RKDT(data=X, levels=self.levels, leafsize=self.leafsize)
                tree.distributed_build()
            else:
                X = copy(self.host_data, self.sparse_flag)
                tree = RKDT(data=X, levels=self.levels, leafsize=self.leafsize)
            timer.pop("Forest: Build Dist Tree")

            #Communicate Distributed Coordinates
            timer.push("Forest: Distribute Coordinates")
            if mpi_size > 1:
                tree.collect_data()
            timer.pop("Forest: Distribute Coordinates")


            timer.push("Forest: Evaluate")
            #TODO: Either support int64 or change to local_ids and update in python
            if self.gpu_flag and (not self.sparse_flag):
                neighbors = Primitives.gpu_dense_knn(tree.global_ids, tree.host_data, tree.local_levels, ltrees, k, blocksize, self.device)
            elif self.gpu_flag and self.sparse_flag:
                neighbors = Primitives.gpu_sparse_knn(tree.global_ids, tree.host_data, tree.local_levels, ltrees, k, blockleaf, blocksize, self.device)
            elif (not self.gpu_flag) and (not self.sparse_flag):
                tree.build_local()
                neighbors = tree.search_local(k)
            elif (not self.gpu_flag) and (self.sparse_flag):
                n, d = tree.host_data.shape
                neighbors = Primitives.cpu_sparse_knn_3(tree.global_ids, tree.host_data.indptr, tree.host_data.indices, tree.host_data.data, len(tree.host_data.data), tree.local_levels, ltrees, k, blocksize, n, d, cores)
            timer.pop("Forest: Evaluate")

            #Sort to check
            #neighbors = Primitives.merge_neighbors(neighbors, neighbors, k)
            #print(rank, "Neighbors before merge: : ", neighbors, flush=True)

            #Redistribute
            timer.push("Forest: Redistribute")
            if mpi_size > 1:
                gids, neighbors = tree.redistribute_results(neighbors)
            timer.pop("Forest: Redistribute")

            #Merge
            timer.push("Forest: Merge")
            if result is None:
                result = Primitives.merge_neighbors(neighbors, neighbors, k, merge_location, cores)
            else:
                result = Primitives.merge_neighbors(result, neighbors, k, merge_location,  cores)
            timer.pop("Forest: Merge")

            timer.push("Forest: Compare")
            if ( truth is not None ) and ( rank == 0 ) :
                rlist, rdist = result
                test = (rlist[:nq, ...], rdist[:nq, ...])

                acc = Primitives.check_accuracy(truth, test)
                Primitives.accuracy.append( acc )

                record.push("Recall", acc[0])
                record.push("Distance", acc[1])

                if verbose:
                    print("Iteration:", it, "Recall:", acc, flush=True)
                if acc[0] > threshold:
                    break_flag = True

            break_flag = self.comm.bcast(break_flag, root=0)
            timer.pop("Forest: Compare")

            if break_flag:
                break

        timer.pop("Forest: Search")
        return result

