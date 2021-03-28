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

    def __init__(self, ntrees=1, levels=0, leafsize=None, data=None, location="CPU", comm=MPI.COMM_WORLD, cores=None):

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
        #TODO: Add error handing for incorrect data type

        self.sparse = sparse_flag

        #Set device id for search kernels
        rank = self.comm.Get_rank()
        self.device = rank%4
        Primitives.set_device(self.device)

        #Setup data
        self.host_data = None
        if self.sparse:
            N, d = data.shape
            local_value = np.asarray(data.data, dtype=np.float32)
            local_rowptr = np.asarray(data.indptr, dtype=self.lprec)
            local_colidx = np.asarray(data.indices, dtype=self.lprec)

            self.host_data = sp.csr_matrix(
                (local_value, local_colidx, local_rowptr), shape=(N, d))
        else:
            self.host_data = np.asarray(data, dtype=np.float32)
       
        self.dim = self.host_data.shape[1] 


    def all_search(self, k, ntrees=5, truth=None):
        timer = Primitives.Profiler()
        record = Primitives.Recorder()
        blocksize = 64
        cores = 4
        Primitives.set_cores(cores)

        result = None 

        rank = self.comm.Get_rank()
        mpi_size = self.comm.Get_size()

        if truth is not None:
            nq = truth[0].shape[0]
            assert(nq == truth[1].shape[0])
            assert(truth[0].shape[1] == truth[1].shape[1])
            assert(truth[0].shape[1]>=k)
        
        for it in range(ntrees):

            timer.push("Build Dist Tree")
            X = np.copy(self.host_data)
            tree = RKDT(data=X, levels=self.levels, leafsize=self.leafsize)
            tree.distributed_build()
            timer.pop("Build Dist Tree")

            print(rank, "GIDS after redistribute ", tree.global_ids, flush=True)

            timer.push("Distribute Coordinates")
            tree.collect_data()
            timer.pop("Distribute Coordinates")
            
            print(rank, "GIDS after collect: ", tree.global_ids, flush=True)
            timer.push("Evaluate")
            #TODO: Either support int64 or change to local_ids and update in python
            print(rank, "local levels", tree.local_levels, flush=True)
            neighbors = Primitives.gpu_dense_knn(tree.global_ids, tree.host_data, tree.local_levels, 2, k, blocksize, self.device)
            timer.pop("Evaluate")
            
            #Sort to check
            #neighbors = Primitives.merge_neighbors(neighbors, neighbors, k)
            print(rank, "Neighbors before merge: : ", neighbors, flush=True)

            timer.push("Forest: Redistribute")
            #Redistribute
            gids, neighbors = tree.redistribute_results(neighbors)
            timer.pop("Forest: Redistribute")

            print(rank, "GIDS after redist", gids, flush=True)
            print(rank, "Result after redistribute:", neighbors, flush=True)

            timer.push("Forest: Merge")
            if result is None:
                result = Primitives.merge_neighbors(neighbors, neighbors, k)
            else:
                result = Primitives.merge_neighbors(result, neighbors, k)
            timer.pop("Forest: Merge")

            print(rank, "Result after merge:", result, flush=True)

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
        



