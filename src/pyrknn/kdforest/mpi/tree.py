import os
from . import error as ErrorType
from . import util as Primitives

from parla import Parla
from parla.array import copy, storage_size
from parla.cpu import cpu
from parla.tasks import *

import time
import os

import numpy as np

if os.environ["PYRKNN_USE_CUDA"] == '1':
    import cupy as cp
else:
    import numpy as cp

import scipy.sparse as sp
import random

from collections import defaultdict, deque
from mpi4py import MPI


from numba import njit, prange

import gc

htable = np.zeros([2**17, 300])


Qlist = []
for k in range(20):
    v = np.random.rand(200, 200)
    v = np.asarray(v, dtype=np.float32, order='F')
    Q = np.linalg.qr(v)[0]
    Qlist.append(Q)

q_idx = 0
ql_idx = 0

@njit(parallel=False)
def collect(lids, gids, mpi_size):
    n = len(lids)
    locations = np.empty(n, dtype=np.int64)

    #loop over gids, specify location
    for i in prange(n):
        locations[i] = int(gids[i]/n)
    reorder_ids = np.argsort(locations, kind='mergesort')

    locations = locations[reorder_ids]
    lids = lids[reorder_ids]

    starts = np.zeros(mpi_size, dtype=np.int32)
    stops =  np.zeros(mpi_size, dtype=np.int32)
    idx = 0
    for i in range(n):
        if i > 0 and locations[i] != locations[i-1]:
            starts[locations[i]] = i
            stops[locations[i-1]] = i
        if i == n-1:
            stops[locations[i]] = i+1

    #stops[mpi_size-1] = n - starts[mpi_size-2]
    return starts, stops-starts, lids

@njit(fastmath=True)
def process_row_before(row, starts, sizes):
    for l in range(len(starts)):
        row[starts[l]:starts[l]+sizes[l]] =  row[starts[l]:starts[l]+sizes[l]] - row[starts[l]]
    return row

@njit(fastmath=True)
def process_row_after(row, starts, sizes):
    for l in range(len(starts)):
        if starts[l] != 0:
            row[starts[l]:starts[l]+sizes[l]] =  row[starts[l]:starts[l]+sizes[l]] + row[starts[l]-1] + 1
    return row

#@njit
def find_index(rows, idx, upper=True):
    if idx == 0:
        return 0

    if idx > rows[len(rows)-1]:
        return len(rows)

    init = np.searchsorted(rows, idx-1)

    i = init
    print(rows[init], flush=True)
    if upper:
        while i < len(rows):
            if rows[i] > rows[init]:
                break
            i = i + 1
    else:
        while i > 0:
            if rows[i] < rows[init]:
                break
            i = i - 1

    return i

class RKDT:

    verbose = False

    def __init__(self, levels=0, leafsize=512, pointset=None, location="CPU", sparse=False, comm=None, N=None, d=None):

        self.id = id(self)

        self.levels = levels
        self.leafsize = leafsize
        self.location = location
        self.sparse = sparse

        self.ordered = False

        if comm is not None:
            self.comm = comm
        else:
            self.comm = MPI.COMM_WORLD

        rank = self.comm.Get_rank()

        self.host_data = None

        if N is not None:
            self.N = N
        if d is not None:
            self.d = d

        if( self.location == "GPU"):
            self.lib = cp
        else:
            self.lib = np

        if (pointset is not None):
            #the number of points in the dataset
            self.local_size = pointset.shape[0]

            #Global IDs of the pointset (assigned in original ordering)
            self.real_gids = np.arange(rank*self.local_size, (rank+1)*self.local_size, dtype=np.int32)
            self.host_real_gids = np.copy(self.real_gids)

            #Local IDs of the pointset (assigned in original ordering)
            self.gids = np.arange(self.local_size, dtype=np.int32)
            self.host_gids = np.copy(self.gids)

            if( self.sparse ):
                #(data, indices, indptr)
                #Copy of data in CPU Memory

                print("Into the Trees", pointset.shape)
                print("N", self.N)
                print("d", self.d)

                local_data = np.asarray(pointset.data, dtype=np.float32)
                local_row = np.asarray(pointset.row, dtype=np.int32)
                local_col = np.asarray(pointset.col, dtype=np.int32)

                self.host_data = sp.coo_matrix( (local_data, (local_row, local_col) ), shape=(self.N, self.d))

            else:
                self.host_data = np.asarray(pointset, dtype=np.float32)

            self.dim = self.host_data.shape[1]

            if (self.local_size == 0):
                self.empty = True
            else:
                self.empty = False

        else:
            self.empty= True
            self.local_size = 0
            self.host_gids = np.array([])
            self.gids = self.lib.asarray([])

            if(self.sparse):
                local_data = np.asarray([], dtype=np.float32)
                local_row = np.asarray([], dtype=np.int32)
                local_col = np.asarray([], dtype=np.float32)

                self.host_data = sp.coo_matrix( (local_data, (local_row, local_col) ))
            else:
                self.host_data = np.asarray([], dtype=np.float32)

        #Assumes all trees have same location and sparsity.
        Primitives.set_env(self.location, self.sparse)

        #Reduce global size of dataset
        temp_local_size = np.array(self.local_size, dtype=np.int32)
        temp_global_size = np.array(0, dtype=np.int32)
        self.comm.Allreduce(temp_local_size, temp_global_size, op=MPI.SUM)
        self.size = temp_global_size

        self.built=False

    def __del__(self):
        del self.vectors
        del self.gids
        del self.data
        del self.host_data

    @classmethod
    def set_verbose(self, v):
        """
        Set mode for print statements and __str__ .

        arguments:
            v -- boolean, verbose or not
        """
        self.verbose = v
        self.Node.verbose = v


    def generate_projection_vectors(self):
        timer = Primitives.Profiler()

        self.dist_levels = int(np.floor(np.log2(self.comm.Get_size())))
        self.levels = int(min(np.ceil(np.log2(np.ceil(self.size/self.leafsize))), self.levels))

        comm = self.comm
        rank = comm.Get_rank()
        size = comm.Get_size()
        global q_idx
        global ql_idx 
        #Generate projection vectors on rank 0
        if rank == 0:
            vectors = np.random.randn(self.dim, self.levels)
            vectors = np.array(vectors, dtype=np.float32, order='F')

            timer.push("Projection: QR")
            #if q_idx < 200-self.levels:
            #    vectors = Qlist[ql_idx][q_idx:q_idx+self.levels, :]
            #    q_idx += self.levels
            #else:
            #    ql_idx += 1
            #    q_idx = 0
            #    vectors = Qlist[ql_idx][q_idx:q_idx+self.levels, :]
            vectors = np.linalg.qr(vectors)[0]
            timer.pop("Projection: QR")

            if self.levels > self.dim:
                spill = np.random.randint(low=0, high=self.dim, size=self.levels-self.dim, dtype=np.int32)
        else:
            ld = min(self.dim, self.levels)
            vectors = np.zeros( (self.dim, ld), dtype=np.float32, order='F')

            if self.levels > self.dim:
                spill = np.zeros(self.levels-self.dim, dtype=np.int32)

        #Broadcast random projections and spillover to all processes
        timer.push("Projection: Bcast")
        self.comm.Bcast(vectors, root=0)
        self.vectors = vectors.T

        if self.levels > self.dim:
            self.comm.Bcast(spill, root=0)
            self.spill = spill
        timer.pop("Projection: Bcast")

    def dist_build(self):
        timer = Primitives.Profiler()

        timer.push("Build: Generate Projection")
        self.generate_projection_vectors()
        timer.pop("Build: Generate Projection")

        if self.sparse:
            self.dist_build_sparse()
        else:
            self.dist_build_dense()

    def dist_build_dense(self):
        timer = Primitives.Profiler()

        comm = self.comm
        rank = comm.Get_rank()
        size = comm.Get_size()

        median_list = []
        prank = comm.Get_rank()

        if size > 1:
            for i in range(self.dist_levels):

                timer.push("Dist Build: Get Global Size")
                rank = comm.Get_rank()
                mpi_size = comm.Get_size()
                global_size = np.array(0, dtype=np.int32)
                local_size = np.array(self.local_size, dtype=np.int32)
                comm.Allreduce(local_size, global_size, op=MPI.SUM)
                timer.pop("Dist Build: Get Global Size")

                timer.push("Dist Build: Compute Projection")
                a = i
                if i >= self.dim:
                    a = int(spill[i-self.dim])

                vector = self.vectors[a, :]
                proj_data = self.host_data @ vector
                timer.pop("Dist Build: Compute Projection")

                timer.push("Dist Build: Distributed Select")
                lids = np.arange(self.local_size, dtype=np.int32)
                median, local_split = Primitives.dist_select(global_size/2, proj_data, lids, comm)
                timer.pop("Dist Build: Distributed Select")


                timer.push("Dist Build: Reorder")
                self.host_real_gids = self.host_real_gids[lids]
                self.host_data = self.reorder(lids, self.host_data)
                timer.pop("Dist Build: Reorder")

                #Redistribute

                timer.push("Dist Build: Compute Targets")
                #Pass local split to rank//2 if rank > comm.Get_size()
                if(rank >= mpi_size//2): #sending small, recv large
                    send_size = local_split
                    send_offset = 0
                    color = 1
                else: #recv small, send large
                    send_size = self.local_size - local_split
                    send_offset = local_split
                    color = 0

                keep_size = self.local_size - send_size
                list_sizes = np.zeros(mpi_size, dtype=np.int32)
                list_sizes = comm.allgather(send_size)

                half = mpi_size//2
                send_dict = defaultdict(list)
                arr = list_sizes

                #roundrobin loop
                for j in range(half):
                    for l in range(half):
                        message_size = min(arr[(l+j)%half+half], arr[l])
                        arr[(l+j)%half+half] = arr[(l+j)%half+half] - message_size
                        arr[l] = arr[l] - message_size
                        tag = j*half+l
                        if message_size > 0:
                            send_dict[l].append( ( (l+j)%half + half, message_size, tag) )

                #Compute incoming
                recv_dict = defaultdict(list)
                for m in send_dict.items():
                    for source in m[1]:
                        recv_dict[source[0]].append( (m[0], source[1], source[2]) )
                if(rank >= mpi_size/2):
                    send_dict = recv_dict
                else:
                    recv_dict = send_dict

                rsizes = np.zeros(mpi_size)
                rsizes[rank] = keep_size
                for m in send_dict[rank]:
                    rsizes[m[0]] = m[1]

                rstarts = np.cumsum(rsizes) - rsizes
                timer.pop("Dist Build: Compute Targets")

                timer.push("Dist Build: AlltoAllv GIDS")

                timer.push("Dist Build: AlltoAllv GIDS: Allocate")
                recv_gids = np.zeros(len(self.host_real_gids), dtype=np.int32)
                timer.pop("Dist Build: AlltoAllv GIDS: Allocate")

                timer.push("Dist Build: AlltoAllv GIDS: Comm")
                comm.Alltoallv([self.host_real_gids, tuple(rsizes), tuple(rstarts), MPI.INT], [recv_gids, tuple(rsizes), tuple(rstarts), MPI.INT])
                timer.pop("Dist Build: AlltoAllv GIDS: Comm")

                timer.pop("Dist Build: AlltoAllv GIDS")

                timer.push("Dist Build: AlltoAllv Data")

                timer.push("Dist Build: AlltoAllv Data: Allocate")
                recv_data = np.zeros(self.host_data.shape, dtype=np.float32)
                timer.pop("Dist Build: AlltoAllv Data: Allocate")

                timer.push("Dist Build: AlltoAllv Data: Comm")
                comm.Alltoallv([self.host_data, tuple(rsizes*self.dim), tuple(rstarts*self.dim), MPI.FLOAT], [recv_data, tuple(rsizes*self.dim), tuple(rstarts*self.dim), MPI.FLOAT])
                timer.pop("Dist Build: AlltoAllv Data: Comm")

                timer.pop("Dist Build: AlltoAllv Data")

                self.host_real_gids = recv_gids
                self.host_data = recv_data

                median_list.append(median)

                #Split communicator
                comm = comm.Split(color, rank)

        self.data = self.host_data
        self.gids = self.host_gids
        self.real_gids = self.host_real_gids

        self.nodelist = [None]
        self.offset_list = None
        self.nodelist[0] = self.Node(self, idx=0, level=0, size=self.local_size, gids=self.gids)

    def dist_build_sparse(self):
        timer = Primitives.Profiler()

        comm = self.comm
        size = comm.Get_size()
        rank = comm.Get_rank()

        median_list = []
        if size > 1:
            for i in range(self.dist_levels):

                timer.push("Dist Build: Get Global Size")
                rank = comm.Get_rank()
                mpi_size = comm.Get_size()
                global_size = np.array(0, dtype=np.int32)
                local_size = np.array(self.local_size, dtype=np.int32)
                comm.Allreduce(local_size, global_size, op=MPI.SUM)
                timer.pop("Dist Build: Get Global Size")

                timer.push("Dist Build: Compute Projection")
                a = i
                if i >= self.dim:
                    a = int(spill[i-self.dim])

                vector = self.vectors[a, :]
                proj_data = self.host_data @ vector
                timer.pop("Dist Build: Compute Projection")

                timer.push("Dist Build: Select")
                lids = np.arange(self.local_size, dtype=np.int32)
                median, local_split = Primitives.dist_select(global_size/2, proj_data, lids, comm)
                timer.pop("Dist Build: Select")

                timer.push("Dist Build: Reorder")
                self.host_real_gids = self.host_real_gids[lids]
                self.host_data = self.reorder(lids, self.host_data)
                timer.pop("Dist Build: Reorder")

                #TODO: Replace this with something saner
                timer.push("Dist Build: Compute Targets")

                #Pass local split to rank//2 if rank > comm.Get_size()
                if(rank >= mpi_size//2): #sending small, recv large
                    send_size = local_split
                    send_offset = 0
                    color = 1
                else: #recv small, send large
                    send_size = self.local_size - local_split
                    send_offset = local_split
                    color = 0

                keep_size = self.local_size - send_size

                list_sizes = np.zeros(mpi_size, dtype=np.int32)
                list_sizes = comm.allgather(send_size)

                half = mpi_size//2
                send_dict = defaultdict(list)
                arr = list_sizes

                #roundrobin loop
                #Compute outgoing
                for j in range(half):
                    for l in range(half):
                        message_size = min(arr[(l+j)%half+half], arr[l])
                        arr[(l+j)%half+half] = arr[(l+j)%half+half] - message_size
                        arr[l] = arr[l] - message_size
                        tag = j*half+l
                        if message_size > 0:
                            send_dict[l].append( ( (l+j)%half + half, message_size, tag) )

                #Compute incoming
                recv_dict = defaultdict(list)
                current_items = send_dict.items()
                for m in current_items:
                    for source in m[1]:
                        recv_dict[source[0]].append( (m[0], source[1], source[2]) )

                if(rank >= mpi_size/2):
                    send_dict = recv_dict
                else:
                    recv_dict = send_dict

                rows    = self.host_data.row
                cols    = self.host_data.col
                data    = self.host_data.data

                #Compute All-Allv Send/Size Row blocks
                #local
                rsizes = [0]*mpi_size
                rsizes[rank] = keep_size
                for m in send_dict[rank]:
                    rsizes[m[0]] = m[1]

                rstarts = np.cumsum(rsizes) - np.array(rsizes)
                timer.pop("Dist Build: Compute Targets")

                timer.push("Dist Build: AlltoAllv GIDS")
                recv_gids = np.zeros(len(self.host_real_gids), dtype=np.int32)
                comm.Alltoallv([self.host_real_gids, tuple(rsizes), tuple(rstarts), MPI.INT],[recv_gids, tuple(rsizes), tuple(rstarts), MPI.INT])
                timer.pop("Dist Build: AlltoAllv GIDS")

                timer.push("Dist Build: Compute NNZ Sends")
                nnzstarts = [0]*mpi_size
                nnzends = [0]*mpi_size

                for l in range(mpi_size):
                    start = find_index(rows, rstarts[l], upper=True)
                    nnzstarts[l] = start

                    end = find_index(rows, rstarts[l]+rsizes[l], upper=True)
                    nnzends[l] = end

                nnzsizes = list(np.array(nnzends) - np.array(nnzstarts))
                all_nnz_sizes = comm.allgather(nnzsizes)

                nnz_recv_sizes = [0]*mpi_size
                for l in range(mpi_size):
                    nnz_recv_sizes[l] = all_nnz_sizes[l][rank]

                nnz_recv_starts = np.cumsum(nnz_recv_sizes) - np.array(nnz_recv_sizes)
                timer.pop("Dist Build: Compute NNZ Sends")


                timer.push("Dist Build: Process Row (Send)")
                rows = process_row_before(rows, np.array(nnzstarts), np.array(nnzsizes))
                timer.pop("Dist Build: Process Row (Send)")

                timer.push("Dist Build: AlltoAllv Row")
                recv_row = np.zeros(np.sum(nnz_recv_sizes), dtype=np.int32)
                comm.Alltoallv([rows, tuple(nnzsizes), tuple(nnzstarts), MPI.INT], [recv_row, tuple(nnz_recv_sizes), tuple(nnz_recv_starts),MPI.INT])
                timer.pop("Dist Build: AlltoAllv Row")

                timer.push("Dist Build: Process Row (Recv)")
                recv_row = process_row_after(recv_row, np.array(nnz_recv_starts), np.array(nnz_recv_sizes))
                timer.pop("Dist Build: Process Row (Recv)")

                timer.push("Dist Build: AlltoAllv Col")
                recv_col = np.zeros(np.sum(nnz_recv_sizes), dtype=np.int32)
                comm.Alltoallv([cols, tuple(nnzsizes), tuple(nnzstarts), MPI.INT], [recv_col, tuple(nnz_recv_sizes), tuple(nnz_recv_starts),MPI.INT])
                timer.pop("Dist Build: AlltoAllv Col")

                timer.push("Dist Build: AlltoAllv Data")
                recv_data = np.zeros(np.sum(nnz_recv_sizes), dtype=np.float32)
                comm.Alltoallv([data, tuple(nnzsizes), tuple(nnzstarts), MPI.FLOAT], [recv_data, tuple(nnz_recv_sizes), tuple(nnz_recv_starts),MPI.FLOAT])
                timer.pop("Dist Build: AlltoAllv Data")


                timer.push("Dist Build: Update Sparse Matrix")
                self.host_real_gids = recv_gids
                self.host_data = sp.coo_matrix( (recv_data, (recv_row, recv_col)), shape=(self.local_size, self.dim) )
                self.host_data.has_canonical_format = True
                timer.pop("Dist Build: Update Sparse Matrix")

                median_list.append(median)

                #Split communicator
                comm = comm.Split(color, rank)

        timer.push("Dist Build: Finalize Sparse Matrix (CSR)")
        rank = self.comm.Get_rank()

        temp = self.host_data.tocsr()

        data = temp.data
        ind = temp.indices
        ptr = temp.indptr

        data = np.asarray(data, dtype=np.float32)
        ind = np.asarray(ind, dtype=np.int32)
        ptr = np.asarray(ptr, dtype=np.int32)

        self.data = sp.csr_matrix((data, ind, ptr), shape=(self.local_size, self.dim))
        self.data.has_sorted_indices = True
        self.gids = np.asarray(self.host_gids, dtype=np.int32)
        self.real_gids = np.asarray(self.host_real_gids, dtype=np.int32)

        timer.pop("Dist Build: Finalize Sparse Matrix (CSR)")

        self.ordered = False
        self.offset_list = None
        self.nodelist = [None]
        self.nodelist[0] = self.Node(self, idx=0, level=0, size=self.local_size, gids=self.gids)


    def reorder(self, lids, data):
        if self.sparse:
            rows = data.row
            cols = data.col
            d = data.data

            #Reassign rows
            labels = np.arange(len(lids), dtype=np.int32)[np.argsort(lids)]
            rows = labels[rows]

            #Reorder matrix
            order = np.argsort(rows, kind='stable')
            rows = rows[order]
            cols = cols[order]
            d = d[order]

            return sp.coo_matrix((d, (rows, cols)))
        else:
            data = data[lids, ...]
            return data

    def hash(self, index, label="id"):
        leaf_nodes = self.get_level(self.levels)
        i = 1
        global htable
        for leaf in leaf_nodes:
            if label == "id":
                htable[leaf.gids, index] = np.ones(len(leaf.gids)) * i
            elif label == "mean":
                htable[leaf.gids, index] = np.ones(len(leaf.gids)) * leaf.center()
            i += 1  

    def build(self, levels=None, leafsize=None):
        timer = Primitives.Profiler()

        rank = self.comm.Get_rank()

        self.levels = int(min( np.ceil(np.log2(np.ceil(self.size/self.leafsize))), self.levels ))
        self.dist_levels = int(np.floor(np.log2(self.comm.Get_size())))

        timer.push("Build: Distributed Build")
        self.dist_build()
        timer.pop("Build: Distributed Build")

        if self.location == "GPU" or self.sparse:
            return

        timer.push("Build: Precompute Offsets")

        self.size_list = [[self.local_size]]
        self.offset_list = [[0]]
        self.host_offset_list = [[0]]

        #Precompute the node sizes at each level
        for level in range(0, self.levels-self.dist_levels+1):
            level_size_list = []
            for n in self.size_list[level]:
                level_size_list.append(np.floor(n/2))
                level_size_list.append(np.ceil(n/2))
            self.size_list.append(level_size_list)

        for level in range(0, self.levels-self.dist_levels+1):
            self.size_list[level].insert(0, 0)

        #Precompute the offset for each node
        for level in range(0, self.levels-self.dist_levels+1):
            self.offset_list.append(np.cumsum(self.size_list[level]))
            self.host_offset_list.append(np.cumsum(self.size_list[level]))

        self.nodelist = [None] * (2**(self.levels - self.dist_levels + 1)-1)

        timer.pop("Build: Precompute Offsets")

        timer.push("Build: Local Build")
        with Parla():
            @spawn(placement = cpu)
            async def build_tree():
                T = TaskSpace()

                #Create the root node
                @spawn(T[0], placement = cpu)
                def create_root():
                    root = self.Node(self, idx=0, level=0, size=self.local_size, gids=self.gids)
                    self.nodelist[0] = root

                await T

                #Build tree in n-order traversal
                for level in range(0, self.levels - self.dist_levels):
                    start = 2**level -1
                    stop  = 2**(level+1) - 1
                    level_size = stop - start
                    data_size = self.local_size/2**level * 4
                    for i in range(level_size):
                        @spawn(T[start+i+1], [T[0], T[int((start+i+1)/2)]], placement = cpu, memory=data_size)
                        def create_children_task():
                            split_t = time.time()
                            current_node = self.nodelist[start+i]
                            if current_node is not None:
                                children = current_node.split()
                                children = list(filter(None, children))
                                for child in children:
                                    idx = child.get_id()
                                    self.nodelist[idx] = child
                            split_t = time.time() - split_t
                await T

        self.built=True
        self.ordered = True

        #Fix overestimate of tree levels (#TODO fix level estimate above)
        self.levels = self.levels-self.dist_levels
        while self.get_level(self.levels)[0] is None:
            self.levels -= 1
            self.nodelist = self.nodelist[:2**(self.levels+1)-1]

        timer.pop("Build: Local Build")


    def redist(self, neighbors):
        timer = Primitives.Profiler()
        timer.push("Redistribute")

        neighbor_ids = neighbors[0]
        neighbor_dist = neighbors[1]

        k = neighbor_ids.shape[1]

        mpi_size = self.comm.Get_size()
        rank = self.comm.Get_rank()
        comm = self.comm

        timer.push("Redistribute: Allocate")
        result_gids = np.zeros(self.local_size, dtype=np.int32)
        result_ids  = np.zeros((self.local_size, k), dtype=np.int32)
        result_dist = np.zeros((self.local_size, k), dtype=np.float32)
        timer.pop("Redistribute: Allocate")

        real_gids = self.host_real_gids
        lids = np.arange(self.local_size, dtype=np.int32)

        timer.push("Redistribute: Compute Targets")
        starts, lengths, lids = collect(lids, real_gids, mpi_size)
        timer.pop("Redistribute: Compute Targets")

        timer.push("Redistribute: Send Targets")
        send_lengths = comm.allgather(lengths)
        recv_lengths = [ list(i) for i in zip(*send_lengths) ]
        timer.pop("Redistribute: Send Targets")

        timer.push("Redistribute: Reorder")
        sending_ids = neighbor_ids[lids, ...]
        sending_dist = neighbor_dist[lids, ...]
        sending_gids = real_gids[lids]
        timer.pop("Redistribute: Reorder")

        ssizes = np.asarray(send_lengths[rank])
        sstarts = np.cumsum(ssizes) - ssizes

        rsizes = np.asarray(recv_lengths[rank])
        rstarts = np.cumsum(rsizes) - rsizes

        timer.push("Redistribute: Alltoall GIDS")
        comm.Alltoallv([sending_gids, tuple(ssizes), tuple(sstarts), MPI.INT], [result_gids, tuple(rsizes), tuple(rstarts), MPI.INT])
        timer.pop("Redistribute: Alltoall GIDS")

        ssizes = ssizes * k
        sstarts = sstarts * k

        rsizes = rsizes * k
        rstarts = rstarts * k

        timer.push("Redistribute: Alltoall IDS")
        comm.Alltoallv([sending_ids, tuple(ssizes), tuple(sstarts), MPI.INT], [result_ids, tuple(rsizes), tuple(rstarts), MPI.INT])
        timer.pop("Redistribute: Alltoall IDS")

        timer.push("Redistribute: Alltoall Dist")
        comm.Alltoallv([sending_dist, tuple(ssizes), tuple(sstarts), MPI.FLOAT], [result_dist, tuple(rsizes), tuple(rstarts), MPI.FLOAT])
        timer.pop("Redistribute: Alltoall Dist")

        timer.push("Redistribute: Sort")
        lids = np.argsort(result_gids)
        new_gids = result_gids[lids]

        result_ids = result_ids[lids, ...]
        result_dist = result_dist[lids, ...]
        timer.pop("Redistribute: Sort")
        timer.pop("Redistribute")

        return new_gids, (result_ids, result_dist)

    class Node:

        verbose = False

        def __init__(self, tree, idx=0, level=0, size=0, gids=None):

            self.tree = tree
            self.id = idx
            self.level = level
            self.size = size

            self.gids = gids
            self.vector = None

            self.anchors = None

            self.isleaf = True
            self.parent = None
            self.children = [None, None]

            if self.tree.offset_list is not None:
                self.offset = int(self.tree.offset_list[self.level+1][idx - 2**self.level + 1])
            else:
                self.offset = 0

            self.lib = self.tree.lib

        def __del__(self):
            del self.gids
            del self.vector

        def __str__(self):
            """Overloading the print function for a RKDT.Node class"""
            if self.verbose:
                msg = 'Node: ' + str(self.id) + ' at level '+ str(self.level) + ' of size ' + str(self.size)
                msg += '\nLeaf:' + str(self.isleaf)
                msg += '\nBelonging to tree with id ' + str(self.tree.id)
                msg += '\nOffset: '+str(self.offset)
                msg += '\nSplitting Line: '+str(self.vector)
                msg += '\nContains gids:' +str(self.gids)
                #msg += '\nData:'+str(self.tree.data[self.gids, ...])
                msg += '\n--------------------------'
            else:
                msg = 'Node: ' + str(self.id) + ' at level '+ str(self.level) + ' of size ' + str(self.size)
            return msg

        def get_id(self):
            return self.id

        def get_gids(self):
            return self.gids

        def center(self):
            data = self.get_reference()
            center = np.mean(data, axes=1)
            return center 

        def get_reference(self):
            if self.tree.ordered:
                return self.tree.data[self.offset:self.offset+self.size, ...]
            else:
                return self.tree.data[self.gids, ...]

        def set_right_child(self, node):
            self.children[0] = node

        def set_left_child(self, node):
            self.children[1] = node

        def set_children(self, nodelist):
            self.children = nodelist
            self.isleaf = True if all([child is None for child in nodelist]) else False

        def set_parent(self, node):
            self.parent = node

        def average(self, idx=0):
            return self.lib.mean(self.tree.host_data[self.gids, idx])

        def median(self, idx=0):
            return self.lib.median(self.tree.host_data[self.gids, idx])

        def select_hyperplane(self):
            a = self.level
            if a+self.tree.dist_levels >= self.tree.dim:
                a = random.randint(0, self.tree.dim-1)
            self.vector = self.tree.vectors[a+self.tree.dist_levels, :]
            self.vector = self.vector / np.linalg.norm(self.vector)
            self.local_ = self.tree.data[self.offset:self.offset+self.size, ...] @ self.vector
            del a

        def cleanup(self):
            del self.local_

        def split(self, stream=None):

            middle = int(self.size//2)
            self.tree.nodelist[self.id] = self

            if (middle < self.tree.leafsize) or (self.level+1) > self.tree.levels:
                self.plane = None
                self.anchors = None
                self.vector = None
                self.isleaf=True
                return [None, None]

            self.ordered = True
            if self.tree.location != "GPU":

                self.select_hyperplane()

                self.lids = self.lib.argpartition(self.local_, middle)  #parition the local ids
                self.tree.gids[self.offset:self.offset+self.size] = self.gids[self.lids]                  #partition the global ids
                self.gids = self.tree.gids[self.offset:self.offset+self.size]
                self.plane = self.local_[self.lids[middle]]       #save the splitting line

                self.cleanup()                                    #delete the local projection (it isn't required any more)


                #Initialize left and right nodes
                left = self.tree.Node(self.tree, level = self.level+1, idx = 2*self.id+1, size=middle, gids=self.gids[:middle])
                right = self.tree.Node(self.tree, level = self.level+1, idx = 2*self.id+2, size=int(self.size - middle), gids=self.gids[middle:])
                left.set_parent(self)
                right.set_parent(self)
                children = [left, right]
                self.set_children(children)

                local_data = self.tree.data[self.offset:self.offset+self.size]
                self.tree.data[self.offset:self.offset+self.size] = local_data[self.lids]

                del local_data
                del self.lids

                return children

            else:

                #project onto line (projection is stored in self.local_)
                with stream:
                    a = self.level - self.tree.dist_levels
                    if a >= self.tree.dim:
                        a = random.randint(0, self.tree.dim-1)

                    self.vector = self.tree.vectors[a, :]

                    self.local_ = self.tree.data[self.offset:self.offset+self.size, ...] @ self.vector
                    self.lids = self.lib.argpartition(self.local_, middle)

                    self.tree.gids[self.offset:self.offset+self.size] = self.gids[self.lids]

                    self.gids = self.tree.gids[self.offset:self.offset+self.size]

                    self.plane = self.local_[self.lids[middle]]
                    local_data = self.tree.data[self.offset:self.offset+self.size]
                    self.tree.data[self.offset:self.offset+self.size] = local_data[self.lids]

                    #del local_data
                    del self.lids
                    del self.local_

                left = self.tree.Node(self.tree, level=self.level+1, idx=2*self.id+1, size=middle, gids=self.gids[:middle])
                right = self.tree.Node(self.tree, level=self.level+1, idx=2*self.id+2, size=int(self.size-middle), gids=self.gids[middle:])

                left.set_parent(self)
                right.set_parent(self)
                children = [left, right]
                self.set_children(children)

                if self.level < 4:
                    stream.synchronize()
                    stream_new = cp.cuda.Stream(non_blocking=True)
                    left.split(stream = stream)
                    right.split(stream = stream_new)
                    stream.synchronize()
                    stream_new.synchronize()
                else:
                    left.split(stream)
                    right.split(stream)

                return children

        def knn(self, Q, k):
            """
            Perform an exact exhaustive knn query search in the node. O(size x gids x d)

            Arguments:
                Q -- N x d matrix of query points
                k -- number of nearest neighbors
            """
            R = self.get_reference()
            print("R shape", R.shape)

            lids = np.arange(self.size, dtype=np.int32)
            print("lids", lids)

            results =  Primitives.single_knn(self.gids, R, Q, k)

            print("Finished Search")

            #Merge with itself to sort output
            results = Primitives.merge_neighbors(results, results, k)

            print("Finished Merge")

            return results


        def knn_all(self, k):
            """
            Perform an exact exhaustive all-knn search in the node. O(size x gids x d)

            Arguments:
                k -- number of nearest neighbors (Limitation: k < leafsize)
            """
            R = self.get_reference()
            results = Primitives.single_knn(self.gids, R, R, k)

            #Merge with itself to sort output
            results = Primitives.merge_neighbors(results, results, k)
            return results

        def single_query(self, q):
            """
            Update the tree index belonging to a single query point.

            i.e. if that query point were added to this node would it have belonged to the left or right child

            Arguments:
                q -- 1 x d query point
            """
            if self.isleaf:
                return self.id

            #compute projection
            if not self.tree.sparse:
                q = q.reshape((1, len(q)))

            dist = q @ self.vector

            #compare against splitting plane
            return 2*self.id+1 if dist < self.plane else 2*self.id+2

    def ordered_data(self):
        if not self.built:
            raise ErrorType.InitializationError('Tree has not been built.')
        if not self.ordered:
            result = self.lib.array(self.data[self.gids, ...], dtype='float32')
            return result
        if self.ordered:
            return self.data

    def cleanup(self):
        self.ordered=False
        del self.data

    def get_levels(self):
        """Return the maximum number of levels in this RKDT."""
        return self.levels

    def get_level(self, level):
        """Return the list of nodes at a level in-order."""
        start = 2**level -1
        stop  = 2**(level+1) - 1
        return self.nodelist[start:stop]

    def single_query(self, q):
        """Find the leaf index corresponding to a single query point.

        Arguments:
            q -- 1 x d query point
        """
        idx = 0
        for l in range(self.levels):
            current_node = self.nodelist[idx]
            idx = current_node.single_query(q)
        return idx

    def query(self, Q):
        """Find the leaf index corresponding to each query point in Q.

        Arguments:
            Q -- N x d query point matrix

        Return value:
            idx -- length N array of node indices
        """

        N = Q.shape[0]
        d = self.dim

        idx = self.lib.zeros(N)

        #TODO: Restructure this for a GPU kernel

        for i in range(N):
            idx = single_query(Q[i, :])
        return idx

    def query_bin(self, Q):
        """Find and bin the leaf index corresponding to each query point in Q""

        Arguments:
            Q -- N x d query point matrix

        Return value:
            bins -- dictionary (key: leaf node index, value: list of local query ids (row idx) for queries in that corresponding leaf)
        """

        #TODO: Restructure this for a multisplit kernel

        N, d, = Q.shape
        bins = defaultdict(list)
        for i in range(N):
            bins[self.single_query(Q[i, :])].append(i)
        return bins

    def knn(self, Q, k):
        """Perform exact exhaustive knn search at the root level of the tree

        Arguments:
            Q -- N x d query point matrix
            k -- The number of nearest neighbors (Require k < leafnodesize, otherwise it is padded with inf values)
        Return:
            (neighbor_list , neigbor_dist)
            neighbor_list -- |Q| x k list of neighbor gids. In local ordering og the query points.
            neighbor_dist -- the corresponding distances
        """
        root = self.nodelist[0]
        return root.knn(Q, k)

    def aknn(self, Q, k):
        """Perform approximate knn search at the root level of the tree

        Arguments:
            Q -- N x d query point matrix
            k -- The number of nearest neighbors (Require k < leafnodesize, otherwise it is padded with inf values)

        Return:
            (neighbor_list , neigbor_dist)
            neighbor_list -- |Q| x k list of neighbor gids. In local ordering og the query points.
            neighbor_dist -- the corresponding distances
        """
        N, d = Q.shape

        #find which leaves belong to which bins
        bins = self.query_bin(Q)
        leaf_keys = [*bins.keys()]

        #Allocate space to store results
        neighbor_list = self.lib.full([N, k], dtype=np.int32)
        neighbor_dist = self.lib.full([N, k], dtype=np.float32)

        #compute batchsize
        MAXBATCH = 1024
        n_leaves = len(bins)
        batchsize = n_leaves if n_leaves < MAXBATCH else MAXBATCH

        #Loop over all batches
        iters = int(np.ceil(n_leaves/batchsize))

        for i in range(iters):
            start = batchsize*(i-1)
            stop  = batchsize*(i) if i < iters-1 else n_leaves

            gidsList = []
            RList = []
            QList = []

            #setup leaf lists
            for leaf in leaf_keys[start:stop]:
                idx = bins[leaf]
                node = self.nodelist[leaf]
                gidsList.append(node.gids)
                RList.append(node.get_reference())
                QList.append(Q[idx, ...])

            #call batch routine
            NLL, NDL = Primitives.multileaf_knn(gidsList, RList, QList, k)

            #populate results from temporary local objects
            j = 0;
            for leaf in leaf_keys[start:stop]:
                idx = bins[leaf]
                node = self.nodelist[leaf]
                lk = min(k, node.size-1)
                NL =  NLL[j]
                ND =  NDL[j]
                neighbor_list[idx, :lk] = NL
                neighbor_dist[idx, :lk] = ND
                j += 1

        return neighbor_list, neighbor_dist

    def aknn_all(self, k):
        """Perform approximate all knn search

        Arguments:
            k -- The number of nearest neighbors (Require k < leafnodesize, otherwise it is padded with inf values)

        Return:
            (neighbor_list , neigbor_dist)
            neighbor_list -- |Q| x k list of neighbor gids. In local ordering og the query points.
            neighbor_dist -- the corresponding distances
        """
        timer = Primitives.Profiler()

        timer.push("AKNN")

        rank = self.comm.Get_rank()

        N = self.local_size
        max_level = self.levels

        #Allocate space to store results
        neighbor_list = self.lib.zeros([N, k], dtype=np.int32)
        neighbor_dist = self.lib.zeros([N, k], dtype=np.float32)

        #get all leaf nodes
        leaf_nodes = self.get_level(max_level)
        n_leaves = len(leaf_nodes)

        real_gids = self.real_gids

        #compute batchsize
        MAXBATCH = 2**28
        n_leaves = len(leaf_nodes)
        batchsize = n_leaves if n_leaves < MAXBATCH else MAXBATCH

        #Loop over all batches
        iters = int(np.ceil(n_leaves/batchsize))

        for i in range(iters):


            timer.push("AKNN: Setup")
            start = batchsize*(i)
            stop  = batchsize*(i+1) if i < iters-1 else n_leaves

            gidsList = []
            RList = []

            #setup leaf lists
            for leaf in leaf_nodes[start:stop]:
                gidsList.append(leaf.gids)
                RList.append(leaf.get_reference())

            timer.pop("AKNN: Setup")

            timer.push("AKNN: Compute")
            NLL, NDL, out = Primitives.multileaf_knn(gidsList, RList, RList, k)
            timer.pop("AKNN: Compute")

            timer.push("AKNN: Copy")

            #populate results from temporary local objects
            #TODO: This can be made much faster

            j = 0;
            for leaf in leaf_nodes[start:stop]:
                idx = leaf.get_gids()
                lk = min(k, leaf.size-1)
                NL =  NLL[j]
                ND =  NDL[j]
                neighbor_list[idx, :] = real_gids[NL[:, :]]
                neighbor_dist[idx, :] = ND[:, :]
                del NL
                del ND
                j += 1

            timer.pop("AKNN: Copy")

            timer.push("AKNN: Cleanup")
            #Clean up
            del NLL
            del NDL
            del out
            del gidsList
            del RList

            #gc.collect()
            timer.pop("AKNN: Cleanup")

        timer.pop("AKNN")

        return neighbor_list, neighbor_dist

    def dist_exact(self, Q, k):
        size = self.comm.Get_size()
        rank = self.comm.Get_rank()

        query_size = Q.shape[0]
        root = self.nodelist[0]

        print("Query_size", Q.shape)

        result =  root.knn(Q, k)

        print("result", result)

        recvbuff_list = np.empty([size, query_size, k], dtype=np.int32)
        recvbuff_dist = np.empty([size, query_size, k], dtype=np.float32)

        rgs = self.real_gids

        ids = rgs[np.asarray(result[0], dtype=np.int32)]

        self.comm.Gather(ids, recvbuff_list, root=0)
        self.comm.Gather(result[1], recvbuff_dist, root=0)

        result = None

        if rank ==0:
            for i in range(size):
                neighbors = (recvbuff_list[i], recvbuff_dist[i])
                if result:
                    result = Primitives.merge_neighbors(result, neighbors, k)
                else:
                    result = neighbors

        return result
