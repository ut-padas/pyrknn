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
import pickle 
import concurrent.futures 

ltrees = 5

def copy(data, sparse_flag):
    if sparse_flag:
        return data.copy()
    else:
        return np.copy(data)

def distributed_tree_task(tree_args):
    t = time.time()
    X, levels, leafsize, rank = tree_args
    t_build = time.time()
    #nX = np.copy(X)
    tree = RKDT(data=X, levels=levels, leafsize=leafsize)
    tree.distributed_build()
    t_build = time.time() - t_build 
    print(rank, "Tree Task Build Time:", t_build, flush=True)

    t_collect = time.time()
    tree.collect_data()
    t_collect = time.time() - t_collect
    print(rank, "Tree Task Collect Time:", t_collect, flush=True)
    t = time.time() - t
    print(rank, "Tree Task Time: ", t, t_build+t_collect, flush=True)
    return tree

def search_task_dense(search_args):
    t = time.time()
    X, gids, k, local_levels, blocksize, device, rank = search_args
    neighbors = Primitives.gpu_dense_knn(gids, X, local_levels, ltrees, k, blocksize,device)
    t = time.time() - t
    print(rank, "Search Task Time: ", t, flush=True)
    return neighbors

def search_task_sparse(search_args):
    t = time.time()
    X, gids, k, local_levels, blocksize, blockleaf, device, rank = search_args 
    neighbors = Primitives.gpu_sparse_knn(tree.global_ids, tree.host_data, tree.local_levels, ltrees, k, blockleaf, blocksize, self.device)
    t = time.time() - t
    print(rank, "Search Task Time: ", t, flush=True)
    return neighbors

class RKDForest:
    """Class for a collection of RKD Trees for Nearest Neighbor searches"""

    verbose = False

    def __init__(self, ntrees=1, levels=0, leafsize=None, data=None, location="HOST", comm=MPI.COMM_WORLD, cores=None):

        #TODO: Make this automatic with psutils
        if cores is None:
            self.cores = 8
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

        self.lprec = np.int32

        #Assign each forest object a unique id
        self.loc_id = id(self)

        #Shared same id as on rank 0
        self.glb_id = 0
        if self.rank == 0:
            self.glb_id = id(self)
        self.glb_id = self.comm.bcast(self.glb_id)

        print("data.shape", data.shape)

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
        #TODO: Add error handing for incorrect data type

        self.sparse_flag = sparse_flag

        self.gpu_flag = (self.location == "GPU")

        #Set device id for search kernels
        rank = self.comm.Get_rank()
        self.device = rank%4
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


    def overlap_search(self, k, ntrees=5, truth=None):
        timer = Primitives.Profiler()
        record = Primitives.Recorder()
        blocksize = 64
        blockleaf = 256
        cores = 56
        Primitives.set_cores(cores)

        result = None 

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
                X = np.copy(self.host_data)
                tree = RKDT(data=X, levels=self.levels, leafsize=self.leafsize)
                tree.distributed_build()
                timer.pop("Build Dist Tree")

                timer.push("Distribute Coordinates")
                tree.collect_data()
                timer.pop("Distribute Coordinates")
                current_tree = tree

            t = time.time()
            with concurrent.futures.ThreadPoolExecutor() as executor:
                tree_args = (self.host_data, self.levels, self.leafsize, rank)
                tree_future = executor.submit(distributed_tree_task, tree_args)

                search_args = (current_tree.host_data, current_tree.global_ids, k, current_tree.local_levels, blocksize, self.device, rank)
                search_future = executor.submit(search_task, search_args)
                next_tree = tree_future.result()
                neighbors = search_future.result()
                #next_tree = tree_future.result()
            t = time.time() - t
            print(rank, "Both Tasks: ", t, flush=True)
            print("NEW TREE: ", next_tree, flush=True)
            #next_tree = None 

            #Sort to check
            #neighbors = Primitives.merge_neighbors(neighbors, neighbors, k)
            #print(rank, "Neighbors before merge: : ", neighbors, flush=True)

            timer.push("Forest: Redistribute")
            print(rank, "Neighbors: ", neighbors.shape, flush=True)
            #Redistribute
            gids, neighbors = current_tree.redistribute_results(neighbors)
            timer.pop("Forest: Redistribute")

            #print(rank, "GIDS after redist", gids, flush=True)
            #print(rank, "Result after redistribute:", neighbors, flush=True)

            timer.push("Forest: Merge")
            if result is None:
                result = Primitives.merge_neighbors(neighbors, neighbors, k)
            else:
                result = Primitives.merge_neighbors(result, neighbors, k)
            timer.pop("Forest: Merge")

            #print(rank, "Result after merge:", result, flush=True)

            current_tree = next_tree

            timer.push("Forest: Compare")
            if ( truth is not None ) and ( rank == 0 ) :
                rlist, rdist = result
                test = (rlist[:nq, ...], rdist[:nq, ...])

                acc = Primitives.accuracy_check(truth, test)
                Primitives.accuracy.append( acc )
                record.push("Recall", acc[0])
                record.push("Distance", acc[1])

                print("Iteration:", it, "Recall:", acc, flush=True)
            timer.pop("Forest: Compare")

        return result        

    def all_search(self, k, ntrees=5, truth=None, cores=56, blocksize=64, blockleaf=128, ltrees=3, threshold=0.95):
        timer = Primitives.Profiler()
        record = Primitives.Recorder()

        timer.push("All Search")
        result = None 

        rank = self.comm.Get_rank()
        mpi_size = self.comm.Get_size()
        break_flag = False 

        if truth is not None:
            nq = truth[0].shape[0]
            assert(nq == truth[1].shape[0])
            assert(truth[0].shape[1] == truth[1].shape[1])
            assert(truth[0].shape[1]>=k)
        
        for it in range(ntrees):

            t = time.time()
            timer.push("Build Dist Tree")
            t_build = time.time()

            if mpi_size > 1:
                X = copy(self.host_data, self.sparse_flag)
                tree = RKDT(data=X, levels=self.levels, leafsize=self.leafsize)
                tree.distributed_build()
            else:
                tree = RKDT(data=self.host_data, levels=self.levels, leafsize=self.leafsize)
            t_build = time.time() - t_build 
            print(rank, "Tree Build:", t_build, flush=True)
            timer.pop("Build Dist Tree")

            #print(rank, "GIDS after redistribute ", tree.global_ids, flush=True)

            timer.push("Distribute Coordinates")
            t_collect = time.time()
            if mpi_size > 1:
                tree.collect_data()
            t_collect = time.time() - t_collect 
            print(rank, "Tree Collect:", t_collect, flush=True)
            timer.pop("Distribute Coordinates")
            t = time.time() - t
            print(rank, "Tree :", t, flush=True)      

            
            #print(rank, "GIDS after collect: ", tree.global_ids, flush=True)
            timer.push("Evaluate")
            t = time.time()

            #TODO: Either support int64 or change to local_ids and update in python
            if self.gpu_flag and (not self.sparse_flag):
                neighbors = Primitives.gpu_dense_knn(tree.global_ids, tree.host_data, tree.local_levels, ltrees, k, blocksize, self.device)
            elif self.gpu_flag and self.sparse_flag:


                print(rank, "DEVICE: ", self.device, flush=True)
                neighbors = Primitives.gpu_sparse_knn(tree.global_ids, tree.host_data, tree.local_levels, ltrees, k, blockleaf, blocksize, self.device)
            elif (not self.gpu_flag) and (not self.sparse_flag):
                #TODO:Need to add this back in 
                assert(False) 
            elif (not self.gpu_flag) and (self.sparse_flag):
                #NOTE: Lots of checks to try to track down segfault. S
                #print(rank, "Running Sparse CPU KNN", tree.global_ids.dtype, tree.ptr, tree.idx, np.max(tree.host_data.indptr), len(tree.host_data.indices), np.min(tree.host_data.indptr), flush=True)
                #if rank == 3 and it == 3:
                #    #time.sleep(1)
                #    with open(f"{2}_csr_{rank}.pickle", 'rb') as f:
                #        #pickle.dump(tree.host_data, f)
                #        compare = pickle.load(f)
                #    X = tree.host_data 
                #    print(rank, "SIdx", np.sum(tree.idx == X.indices), len(X.indices), X.indices.flags, tree.idx.flags, flush=True)
                #    print(rank, "SPtr", np.sum(tree.ptr == X.indptr), len(X.indptr), X.indptr.flags, tree.ptr.flags, flush=True)
                #    print(rank, "SVal", np.sum(tree.val == X.data), len(X.data), X.data.flags, tree.val.flags, flush=True)

                #    print(rank, "Idx", np.sum(compare.indices == X.indices), len(X.indices), len(compare.indices), X.indices.flags, compare.indices.flags, flush=True)
                #    print(rank, "Ptr", np.sum(compare.indptr == X.indptr), len(X.indptr), len(compare.indptr), X.indptr.flags, compare.indptr.flags, flush=True)
                #    print(rank, "Val", np.sum(compare.data == X.data), len(X.data), len(compare.data), X.data.flags, compare.data.flags, flush=True)
                #    print(rank, "os pid", os.getpid(), (compare != X).nnz, flush=True)
                #    print(rank, "datatype compare: ", compare.indices.dtype, X.indices.dtype, compare.indptr.dtype, X.indptr.dtype, compare.data.dtype, X.data.dtype, flush=True)
                #    print(rank, "shape", compare.shape, X.shape)

                #    tree.host_data = compare

                #Lets force this to work. 
                print(rank, "Running on CPU", flush=True)
                flag = True
                while(flag):
                    try:
                        n, d = tree.host_data.shape 
                        #neighbors = Primitives.cpu_sparse_knn(tree.global_ids, tree.host_data, tree.local_levels, ltrees, k, blocksize, cores)
                        neighbors = Primitives.cpu_sparse_knn_3(tree.global_ids, tree.host_data.indptr, tree.host_data.indices, tree.host_data.data, len(tree.host_data.data), tree.local_levels, ltrees, k, blocksize, n, d, cores)
                        #neighbors = Primitives.cpu_sparse_knn_3(tree.global_ids, tree.ptr, tree.idx, tree.val, len(tree.val), tree.local_levels, ltrees, k, blocksize, n, d, cores)
                        #print(rank, "FINISHED SPARSE CPU KNN", flush=True)
                        flag = False
                    except:
                        print(rank, "reloading", flush=True)
                        with open(f"{3}_csr_{rank}.pickle", 'rb') as f:
                            pickle.dump(tree.host_data, f)
                        with open(f"{3}_csr_{rank}.pickle", 'wb') as f:
                            compare = pickle.load(f)
                        tree.host_data = compare
            t = time.time() - t 
            timer.pop("Evaluate")
            print(rank, "Search :", t, flush=True)
            
            #Sort to check
            #neighbors = Primitives.merge_neighbors(neighbors, neighbors, k)
            #print(rank, "Neighbors before merge: : ", neighbors, flush=True)

            timer.push("Forest: Redistribute")
            #Redistribute
            if mpi_size > 1:
                gids, neighbors = tree.redistribute_results(neighbors)
            timer.pop("Forest: Redistribute")

            #print(rank, "GIDS after redist", gids, flush=True)
            #print(rank, "Result after redistribute:", neighbors, flush=True)

            timer.push("Forest: Merge")
            if result is None:
                result = Primitives.merge_neighbors(neighbors, neighbors, k, self.location, cores)
            else:
                result = Primitives.merge_neighbors(result, neighbors, k, self.location,  cores)
            timer.pop("Forest: Merge")

            #print(rank, "Result after merge:", result, flush=True)

            timer.push("Forest: Compare")
            if ( truth is not None ) and ( rank == 0 ) :
                rlist, rdist = result
                test = (rlist[:nq, ...], rdist[:nq, ...])

                acc = Primitives.accuracy_check(truth, test)
                Primitives.accuracy.append( acc )
                record.push("Recall", acc[0])
                record.push("Distance", acc[1])

                print("Iteration:", it, "Recall:", acc, flush=True)
                if acc[0] > threshold:
                    break_flag = True
            break_flag = self.comm.bcast(break_flag, root=0) 
            timer.pop("Forest: Compare")

            if break_flag:
                break

        timer.pop("All Search")
        return result
        



