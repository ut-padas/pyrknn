import os
print(os.environ["PRKNN_USE_CUDA"])
print(bool(os.environ["PRKNN_USE_CUDA"]))

from . import error as ErrorType
from . import util as Primitives

from parla import Parla
from parla.array import copy, storage_size
from parla.cpu import cpu
from parla.tasks import *

import time
import os

import numpy as np

if os.environ["PRKNN_USE_CUDA"] == '1':
    import cupy as cp
else:
    import numpy as cp

import scipy.sparse as sp
import random
 
from collections import defaultdict, deque
from mpi4py import MPI


from numba import njit, prange

import gc


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

    print("reordered locations", locations)

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

class RKDT:
    """Class for Randomized KD Tree Nearest Neighbor Searches"""

    verbose = False #Note: This is a static variable shared by all instances

    def __init__(self, levels=0, leafsize=512, pointset=None, location="CPU", sparse=False, comm=None):
        """Initialize  Randomized KD Tree

            Keyword arguments:
                levels -- maximum number of levels in the tree
                leafsize -- Leaves of size (2*leafsize + 1) will not be split
                pointset -- the N x d dataset of N d dimensional points in Rn
        """

        self.id = id(self)                      #unique id for instance of this class
        self.levels = levels
        self.leafsize = leafsize
        self.location = location
        self.sparse = sparse 

        self.ordered = False  #Is self.data stored in a tree order in memory

        if comm is not None:
            self.comm = comm
        else:
            self.comm = MPI.COMM_WORLD

        rank = self.comm.Get_rank()
        #self.loaded_data = False   #Does self.data exist?
        #self.loaded_query = False  #Does self.projection exist?

        #self.lowmem = False   #Is self.projection 
        #self.batch_levels = self.levels

        #self.data_offset = 0

        self.host_data = None
        
        if(self.location == "CPU"):
            self.lib = np
        elif(self.location == "GPU"):
            self.lib = cp
            #cp.cuda.runtime.setDevice(rank%4)

        if (pointset is not None):
            self.local_size = pointset.shape[0]           #the number of points in the pointset
            
            self.real_gids = np.arange(rank*self.local_size, (rank+1)*self.local_size, dtype=np.int32)    #the global ids of the points in the pointset (assign original ordering)
            self.gids = np.arange(self.local_size, dtype=np.int32)

            #print(rank, "init", self.real_gids, self.gids)

            self.host_gids = np.arange(self.local_size, dtype=np.int32)
            self.host_real_gids = np.arange(rank*self.local_size, (rank+1)*self.local_size, dtype=np.int32)

            if( self.sparse ):
                #(data, indices, indptr)

                #Copy of data in CPU Memory

                local_data = np.asarray(pointset.data, dtype=np.float32)
                local_row = np.asarray(pointset.row, dtype=np.int32)
                local_col = np.asarray(pointset.col, dtype=np.int32)

                self.host_data = sp.coo_matrix( (local_data, (local_row, local_col) ))

                #local_data = self.lib.asnumpy(pointset.data)
                #local_indices = self.lib.asnumpy(pointset.indices)
                #local_indptr = self.lib.asnumpy(pointset.indptr)

                #self.host_data = sp.csr_matrix( (local_data, local_indicies, local_indptr) )

                #Copy of data in location Memory
                #local_data = self.lib.asarray(pointset.data, dtype=np.float32)
                #local_indices = self.lib.asarray(pointset.indices, dtype=np.int32)
                #local_indptr = self.lib.asarray(pointset.indptr, dtype=np.int32)
                #self.data = self.lib.csr_matrix( (local_data, local_indicies, local_indptr) )

            else:
                #Copy of data in CPU Memory
                #print(type(pointset))
                self.host_data = np.array(pointset, dtype=np.float32)
                ##Copy of data in location memory
                #self.data = self.lib.asarray(pointset, dtype=np.float32)



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

                #Copy of CPU Memory
                local_data = np.asarray([], dtype=np.float32)
                local_row = np.asarray([], dtype=np.int32)
                local_col = np.asarray([], dtype=np.float32)
                
                self.host_data = sp.coo_matrix( (local_data, (local_row, local_col) ))

                #Copy of location memory
                #local_data = self.lib.asarray([], dtype=np.float32)
                #local_indices = self.lib.asarray([], dtype=np.int32)
                #local_indptr = self.lib.asarray([], dtype=np.float32)
                
                #self.data = self.lib.csr_matrix( (local_data, local_indicies, local_indptr) )
            else:
                #Copy of data in CPU Memory
                self.host_data = np.asarray([], dtype=np.float32)
                #Copy of data in location memory
                #self.data = self.lib.asarray([], dtype=np.float32)
                if(self.location == "CPU"): 
                    self.data = self.host_data

        #Assumes all trees have the same location and (dense/sparse)ness        
        Primitives.set_env(self.location, self.sparse)

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


    def dist_build_lil(self):

        #dist_t = time.time()
        comm = self.comm
        rank = comm.Get_rank()
        
        self.dist_levels = int(np.floor(np.log2(self.comm.Get_size())))
 
        #Create and share projection vectors
        if rank == 0:
            
            vectors = np.random.rand(self.dim, self.dist_levels)
            vectors = np.array(vectors, dtype=np.float32)

            qr_t = time.time()
            vectors = np.linalg.qr(vectors)[0]
            qr_t = time.time() - qr_t

            print("QR Time", qr_t)
            print(rank, vectors.shape)

            if self.dist_levels > self.dim:
                spill = np.random.randint(low=0, high=self.dim, size=self.dist_levels-self.dim)

        else:
            ld = min(self.dim, self.dist_levels)
            vectors = np.zeros( (self.dim, ld), dtype=np.float32)
            if self.dist_levels > self.dim:
                spill = np.zeros(self.dist_levels-self.dim, dtype=np.int64)

        self.comm.Bcast(vectors, root=0)
        if self.dist_levels > self.dim:
            self.comm.Bcast(spill, root=0)

        vectors = vectors.T
        self.vectors = vectors

        if self.sparse:
            self.host_data = self.host_data.tolil()

        self.dist_vectors = vectors[:self.dist_levels, :]

        #Distributed Level by Level construction
        #GOFMM Style (only support p = 2^x)
        comm = self.comm
        median_list = []

        collect_t_f = 0
        wait_t_f = 0
        copy_t_f = 0
        #print(rank, "Constructing", self.dist_levels, flush=True)
        for i in range(self.dist_levels):

            #Project and Reorder

            #Update from subcommunicator
            rank = comm.Get_rank()
            mpi_size = comm.Get_size()

            collect_t = time.time()

            global_size = np.array(0, dtype=np.int32)
            local_size = np.array(self.local_size, dtype=np.int32)

            comm.Allreduce(local_size, global_size, op=MPI.SUM)

            #print(rank, "gs", global_size, flush=True)

            a = i
            if i >= self.dim:
                a = int(spill[i-self.dim])

            vector = self.dist_vectors[a, :]

            proj_data = self.host_data @ vector

            lids = np.arange(self.local_size, dtype=np.int32)
            #print(rank, global_size/2, proj_data.shape, lids.shape, flush=True)
            median, local_split = Primitives.dist_select(global_size/2, proj_data, lids, comm)

            print(rank, "med", median, flush=True)
            print(rank, "split", local_split=True)

            

            self.host_real_gids = self.host_real_gids[lids] 
            self.host_data = self.host_data[lids, ...]

            #self.host_data = self.reorder(lids, self.host_data)

            #print(rank, "original_proj", proj_data)

            print(rank, "max_o_p", np.max(proj_data))
            print(rank, "min_o_p", np.min(proj_data))
            #print(rank, "local_median", np.median(proj_data))
            #Redistribute 


            #Pass local split to rank//2 if rank > comm.Get_size()
            if(rank >= mpi_size//2): #sending small, recv large
                send_size = local_split
                send_offset = 0
                color = 1
            else: #recv small, send large
                send_size = self.local_size - local_split
                send_offset = local_split
                color = 0


            list_sizes = np.zeros(mpi_size, dtype=np.int32)
            list_sizes = comm.allgather(send_size)
            #print(rank, "list_sizes", list_sizes)
           
            #An unfortunatly O(p^2) load balancing computation
            half = mpi_size//2
            send_dict = defaultdict(list)
            arr = list_sizes

            #roundrobin loop
            for j in range(half):
                for i in range(half):
                    #Compute max message size
                    #print("Edge:", (i, (i+j)%4 + half))
                    #print("State:", arr)
                    message_size = min(arr[(i+j)%half+half], arr[i])
                    arr[(i+j)%half+half] = arr[(i+j)%half+half] - message_size
                    arr[i] = arr[i] - message_size
                    tag = j*half+i
                    if message_size > 0:
                        send_dict[i].append( ( (i+j)%half + half, message_size, tag) )
                    #print("Update:", arr)

            #Compute incoming
            recv_dict = defaultdict(list)
            for m in send_dict.items():
                for source in m[1]:
                    recv_dict[source[0]].append( (m[0], source[1], source[2]) )

            if(rank >= mpi_size/2):
                send_dict = recv_dict
            else:
                recv_dict = send_dict

            #print(rank, "Send Dict", send_dict, flush=True)
            #print(rank, "Recv Dict", recv_dict, flush=True)
            collect_t_f += time.time() - collect_t

            MAX_SIZE=2**24

            blocksize = int( MAX_SIZE / self.dim)
            #print(rank, blocksize)
            #if self.sparse:
            #    blocksize = int(MAX_SIZE)

            #Grab views of what will be for redistributed
            sending_ids = self.host_real_gids[send_offset:send_offset+send_size]


            sending_data = self.host_data[send_offset:send_offset+send_size, ...]

            #print(rank, "median", median, flush=True)
            #print(rank, "max_proj_before", np.max(sending_data @ vector), flush=True)
            #print(rank, "min_pro_before", np.min(sending_data @ vector), flush=True)

            #Loop over nonblocking sends
            send_id_requests = deque()
            send_data_requests = deque()

            offset = 0
            for message in send_dict[rank]:
                message_size = message[1]
                target = message[0]
                tag = message[2]

                #Create local views
                send_id = sending_ids[offset:offset+message_size]
                send_data = sending_data[offset:offset+message_size, ...]

                nblocks = message_size // blocksize
                iters = int(np.ceil(message_size / blocksize))

                #print(rank, "Creating send: ", rank, " -> ", target, " tag: ", tag, "size:", 4*self.dim*message_size, "size:", self.host_gids.itemsize*message_size, flush=True)

                block_offset = 0
                for j in range(iters):
                    if j == iters-1:
                        blocksize = message_size % blocksize 
                                            
                    t = j + 1

                    #Send IDS

                    #print(rank, "Creating send: ", rank, " -> ", target, " tag: ", t*mpi_size+tag+1, "size:", 4*self.dim*message_size, "size:", self.host_gids.itemsize*message_size, flush=True)
                    send_id_block = send_id[block_offset:block_offset+blocksize]
                    temp_id_req = comm.isend(send_id_block, dest=target, tag=t*mpi_size+tag+1)
                    send_id_requests.append(temp_id_req)
                    
                    #Send Data
                    send_data_block = send_data[block_offset:block_offset+blocksize, ...]
                    #print(rank, "send", mpi_size, nblocks, t, tag, flush=True)
                    #print(rank, "Creating send: ", rank, " -> ", target, " tag: ", mpi_size*(nblocks+1) + t*mpi_size+tag+1, "size:", 4*self.dim*blocksize, "size:", self.host_gids.itemsize*message_size, flush=True)
                    temp_data_req = comm.isend(send_data_block, dest=target, tag=mpi_size*(nblocks+1) + t*mpi_size + tag+1)
                    send_data_requests.append(temp_data_req)

                    block_offset += blocksize
                offset += message_size

            #print(rank, "Finished queueing sends")

            blocksize = int( MAX_SIZE / self.dim)
            #if self.sparse:
            #    blocksize = int(MAX_SIZE)

            recv_id_requests = deque()
            recv_data_requests = deque()

            for message in recv_dict[rank]:

                message_size = message[1]
                source = message[0]
                tag = message[2]

                nblocks = message_size // blocksize
                iters = int(np.ceil(message_size / blocksize))

                #print(rank, "Creating recv: ", rank, " <- ", source, " tag: ", tag, "size:", self.host_data.itemsize*self.dim*message_size, "size:", self.host_gids.itemsize*message_size, flush=True)

                for j in range(iters): 
                    t = j + 1

                    #print(rank, 10*4*self.dim*blocksize+10, flush=True) 
                    #print(rank, "Creating recv: ", rank, " <- ", source, " tag: ", t*mpi_size+tag+1, "size:", 4*self.dim*blocksize, "size:", self.host_gids.itemsize*message_size, flush=True)

                    #Recv IDS
                    temp_id_req = comm.irecv(10*self.host_gids.itemsize*blocksize + 10, source=source, tag=t*mpi_size+tag+1)
                    recv_id_requests.append(temp_id_req)
                                       
                    #Recv Data
                    if self.sparse:
                        #print(rank, "recv", mpi_size, nblocks, t, tag, flush=True)
                        #print(rank, "Creating recv: ", rank, " <- ", source, " tag: ", mpi_size*(nblocks+1) + t*mpi_size+tag+1, "size:", 4*self.dim*message_size, "size:", self.host_gids.itemsize*message_size, flush=True)
                        temp_data_req = comm.irecv(10*4*self.dim*blocksize + 10, source=source, tag=mpi_size*(nblocks+1) + t*mpi_size+tag+1)
                        recv_data_requests.append(temp_data_req)

                    else:
                        #print(rank, "recv", mpi_size, nblocks, t, tag, flush=True)
                        #print(rank, "Creating recv: ", rank, " <- ", source, " tag: ", mpi_size*(nblocks+1) + t*mpi_size+tag+1, "size:", self.host_data.itemsize*self.dim*message_size, "size:", self.host_gids.itemsize*message_size, flush=True)
                        temp_data_req = comm.irecv(10*self.host_data.itemsize*self.dim*blocksize + 10, source=source, tag=mpi_size*(nblocks+1) + t*mpi_size+tag+1)
                        recv_data_requests.append(temp_data_req)

            #Alternate sends and recvs from high/low

            #Get maximum number of sends
            local_sends = len(send_id_requests)
            local_sends = np.array(local_sends, dtype=np.int32)

            max_sends = np.array(0, dtype=np.int32)
            comm.Allreduce(local_sends, max_sends, op=MPI.MAX)
            
            id_offset = 0
            data_offset = 0

            for i in range(2*max_sends):
                active = False
                if i%2 == 0:
                    if rank >= half:
                        active = True
                else:
                    if rank < half:
                        active = True

                if active:
                    #Send data to other half
                    #print(rank, "sending", flush=True)
                    if len(send_id_requests) > 0:
                        #print(rank, "s wait id", flush=True) 
                        req = send_id_requests.popleft()
                        req.wait()

                    if len(send_data_requests) > 0:
                        #print(rank, "s wait dat", flush=True)
                        req = send_data_requests.popleft()
                        req.wait()

                else:
                    #Recv data from other half
                    #print(rank, "recving", flush=True)
                    if len(recv_id_requests) > 0:
                        #print(rank, "wait id", flush=True)
                        req = recv_id_requests.popleft()
                        new_id = req.wait()
                        message_size = new_id.shape[0]
                        sending_ids[id_offset:id_offset+message_size] = new_id
                        id_offset += message_size
                    
                    if len(recv_data_requests) > 0:
                        #print(rank, "wait data", flush=True)
                        req = recv_data_requests.popleft()
                        new_data = req.wait()
                        #print("new_data-shape", new_data.shape, flush=True)
                        message_size = new_data.shape[0]
                        sending_data[data_offset:data_offset+message_size, ...] = new_data
                        data_offset += message_size

            comm.Barrier()
            
            self.host_data[send_offset:send_offset+send_size, ...] = sending_data
            self.host_real_gids[send_offset:send_offset+send_size] = sending_ids

            #print(rank, "median", median, flush=True)
            #print(rank, "max_proj_after", np.max(sending_data @ vector), flush=True)
            #print(rank, "min_proj_after", np.min(sending_data @ vector), flush=True)
            
            median_list.append(median)           

            #Split communicator
            comm = comm.Split(color, rank)


        #dist_t = time.time() - dist_t
        #print("Distributed Build Time:", dist_t)
        #print("build dist copy_t", copy_t_f)
        #print("build dist collect_t", collect_t_f)

        if self.sparse:
            temp = self.host_data.tocsr()
            data = temp.data
            ind  = temp.indices
            ptr  = temp.indptr
            
            #Convert to correct datatype and location
            data = self.lib.array(data, dtype=np.float32)
            ind  = self.lib.array(ind, dtype=np.int32)
            ptr  = self.lib.array(ptr, dtype=np.int32)

            if self.location == "CPU":
                self.data = sp.csr_matrix((data, ind, ptr))
            else:
                self.data = self.lib.csr_matrix((data, ind, ptr))
        else:
            self.data = self.lib.array(self.host_data)

        self.gids = self.lib.array(self.host_gids)
        self.real_gids = self.lib.array(self.host_real_gids)


    def generate_projection_vectors(self):
        timer = Primitives.Profiler()

        comm = self.comm
        rank = comm.Get_rank()
        size = comm.Get_size()

        #Generate projection vectors on rank 0
        if rank == 0:
            vectors = np.random.rand(self.dim, self.levels)
            vectors = np.array(vectors, dtype=np.float32, order='F')

            vectors = np.linalg.qr(vectors)[0]

            if self.levels > self.dim:
                spill = np.random.randint(low=0, high=self.dim, size=self.levels-self.dim, dtype=np.int32)
        else:
            ld = min(self.dim, self.levels)
            vectors = np.zeros( (self.dim, ld), dtype=np.float32, order='F')

            if self.levels > self.dim:
                spill = np.zeros(self.levels-self.dim, dtype=np.int32)

        #Broadcast random projections and spillover to all processes
        self.comm.Bcast(vectors, root=0)
        self.vectors = vectors.T

        if self.levels > self.dim:
            self.comm.Bcast(spill, root=0)
            self.spill = spill

    def dist_build(self):

        timer = Primitives.Profiler()

        comm = self.comm
        rank = comm.Get_rank()
        size = comm.Get_size()

        timer.push("Dist Build: Generate Ortho Transform")
        self.generate_projection_vectors()
        timer.pop("Dist Build: Generate Ortho Transform")

        self.dist_levels = int(np.floor(np.log2(self.comm.Get_size())))
        self.levels = int(min( np.ceil(np.log2(np.ceil(self.size/self.leafsize))), self.levels ))
        if self.levels < self.dist_levels:
            self.levels = self.dist_levels
        
        median_list = []

        prank = comm.Get_rank()

        collect_t_f = 0
        wait_t_f = 0
        copy_t_f = 0

        if size > 1:
            #print(rank, "Constructing", self.dist_levels, flush=True)
            for i in range(self.dist_levels):

                #Project and Reorder
                #Update from subcommunicator

                timer.push("Dist Build: Get Global Size")
                rank = comm.Get_rank()
                mpi_size = comm.Get_size()

                global_size = np.array(0, dtype=np.int32)
                local_size = np.array(self.local_size, dtype=np.int32)

                comm.Allreduce(local_size, global_size, op=MPI.SUM)
                timer.pop("Dist Build: Get Global Size")

                #print(rank, "gs", global_size, flush=True)

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
                #print(rank, "list_sizes", list_sizes)
               
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

                comm.Alltoallv([self.host_real_gids, tuple(rsizes), tuple(rstarts), MPI.INT], [recv_gids, tuple(rsizes), tuple(rstarts), MPI.INT])
                timer.pop("Dist Build: AlltoAllv GIDS")

                timer.push("Dist Build: AlltoAllv Data")

                timer.push("Dist Build: AlltoAllv Data: Allocate")
                recv_data = np.zeros(self.host_data.shape, dtype=np.float32)
                timer.pop("Dist Build: AlltoAllv Data: Allocate")

                comm.Alltoallv([self.host_data, tuple(rsizes*self.dim), tuple(rstarts*self.dim), MPI.FLOAT], [recv_data, tuple(rsizes*self.dim), tuple(rstarts*self.dim), MPI.FLOAT])
                timer.pop("Dist Build: AlltoAllv Data")

                self.host_real_gids = recv_gids
                self.host_data = recv_data

                median_list.append(median)           

                #Split communicator
                comm = comm.Split(color, rank)


        self.data = self.host_data
        self.gids = self.host_gids
        self.real_gids = self.host_real_gids


    def reorder(self, lids, data):
        if self.sparse:
            rows = data.row
            cols = data.col
            d = data.data

            #Reassign rows
            rows = lids[rows]

            #Reorder matrix
            order = np.argsort(rows, kind='stable')
            rows = rows[order]
            cols = cols[order]
            d = d[order]

            return sp.coo_matrix((d, (rows, cols)))         
        else:
            data = data[lids, ...]
            return data

    def build(self, levels=None, leafsize=None):
        """
        Construct the RKD Tree

        Keyword Arguments:
            levels -- maximum number of levels in the tree
            leafsize -- Leaves of size(2*leafsize+1) will not be split
        """
        #Various error checking methods to make sure tree is initialized properly
        if self.built:
            raise ErrorType.InitializationError('You cannot call build on a tree that has already been built')
        if self.empty:
            raise ErrorType.InitializationError('Cannot build an empty tree')
        if self.size < 0:
            raise ErrorType.InitializationError('Invalid size parameter: Cannot build a tree of size '+str(self.size))
        if self.leafsize <= 0:
            raise ErrorType.InitializationError('Invalid leaf size parameter: Cannot build a tree with leaf size '+str(self.leafsize))
        if self.levels < 0:
            raise ErrorType.InitializationError('Invalid max levels parameter: Cannot build a tree of '+str(self.levels)+' levels')


        timer = Primitives.Profiler()

        rank = self.comm.Get_rank()

        #Find out the maximum number of levels required
        #TODO: Fix definition of leafsize to make this proper. Can often overestimate level by one.
        self.levels = int(min( np.ceil(np.log2(np.ceil(self.size/self.leafsize))), self.levels ))
        self.dist_levels = int(np.floor(np.log2(self.comm.Get_size())))
        
        timer.push("Build: Distributed Build")
        self.dist_build()
        timer.pop("Build: Distributed Build")

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
        if self.location == "CPU":
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
        
        elif self.location == "GPU":
            print("Running GPU Build")
            root = self.Node(self, idx=0, level=0, size=self.local_size, gids=self.gids)
            self.root = root
            root.split(cp.cuda.Stream(non_blocking=True))
            self.built=True

        self.ordered = True
       
        #Fix overestimate of tree levels (see #TODO above)
        self.levels = self.levels-self.dist_levels
        while self.get_level(self.levels)[0] is None:
            self.levels -= 1
            self.nodelist = self.nodelist[:2**(self.levels+1)-1]

        timer.pop("Build: Local Build")


    def redist(self, neighbors):
        #print("Entering Redistribute", flush=True)

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

        #print(rank, "self.real_gids", self.real_gids)

        real_gids = self.host_real_gids
        #print(rank, "Real_gids", real_gids, flush=True)

        lids = np.arange(self.local_size, dtype=np.int32)
        
        timer.push("Redistribute: Compute Targets")
        starts, lengths, lids = collect(lids, real_gids, mpi_size)
        timer.pop("Redistribute: Compute Targets")

        timer.push("Redistribute: Send Targets")
        send_lengths = comm.allgather(lengths)
        recv_lengths = [ list(i) for i in zip(*send_lengths) ]

        timer.pop("Redistribute: Send Targets")

        #print("SL", send_lengths, flush=True)
        #print("RL", recv_lengths, flush=True)

        timer.push("Redistribute: Reorder")
        sending_ids = neighbor_ids[lids, ...]
        sending_dist = neighbor_dist[lids, ...]
        sending_gids = real_gids[lids]
        timer.pop("Redistribute: Reorder")

        ssizes = np.asarray(send_lengths[rank])
        sstarts = np.cumsum(ssizes) - ssizes

        rsizes = np.asarray(recv_lengths[rank])
        rstarts = np.cumsum(rsizes) - rsizes

        #print(rank, "rsizes", rsizes, flush=True)
        #print(rank, "rstarts", rstarts, flush=True)
        #print(rank, "ssizes", ssizes, flush=True)
        #print(rank, "sstarts", sstarts, flush=True)

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
        #print(rank, "new_gids", new_gids, flush=True)
        timer.pop("Redistribute")

        return new_gids, (result_ids, result_dist)
 

        


    def redist_old(self, neighbors):

        timer = Primitives.Profiler()

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

        timer.push("Redistribute: Compute Targets")
        if self.location == "GPU":
            real_gids = self.host_real_gids
        else:
            real_gids = self.real_gids

        lids = np.arange(self.local_size, dtype=np.int32)

        starts, lengths, lids = collect(lids, real_gids, mpi_size)

        all_lengths = []
        for i in range(mpi_size):
            if i == rank:
                data = lengths
            else:
                data = None
            data = self.comm.bcast(data, root=i)
            all_lengths.append(data)

        timer.pop("Redistribute: Compute Targets")

        timer.push("Redistribute: Reorder")
        sending_ids = neighbor_ids[lids, ...]
        sending_dist = neighbor_dist[lids, ...]
        sending_gids = real_gids[lids]
        timer.pop("Redistribute: Reorder")

        offset = 0
        comm_t = time.time()
        for r in range(mpi_size):

            split_sizes = all_lengths[r]
            split_starts = starts

            recv_size = split_sizes[rank]

            if r == rank:
                send_block_gids = sending_gids
                send_block_ids = sending_ids
                send_block_dist = sending_dist
                sstarts = tuple(split_starts)
                ssizes  = tuple(split_sizes)

            else:
                send_block_gids = None
                send_block_ids = None
                send_block_dist = None
                ssizes = None
                sstarts = None

            #communicate gids
            timer.push("Redistribute: Scatter GID")

            timer.push("Redistribute: Scatter GID: Allocate")
            recv_gid = np.empty(recv_size, dtype=np.int32)
            timer.pop("Redistribute: Scatter GID: Allocate")

            timer.push("Redistribute: Scatter GID: Comm")
            comm.Scatterv([send_block_gids, ssizes, sstarts, MPI.INT], recv_gid, root=r)
            timer.pop("Redistribute: Scatter GID: Comm")

            timer.push("Redistribute: Scatter GID: Copy")
            result_gids[offset:offset+recv_size, ...] = recv_gid
            timer.pop("Redistribute: Scatter GID: Copy")

            timer.pop("Redistribute: Scatter GID")

            split_sizes = split_sizes * k
            split_starts = starts * k

            if r == rank:
                sstarts = tuple(split_starts)
                ssizes  = tuple(split_sizes)

            else:
                ssizes = None
                sstarts = None

            #communicate ID block
            timer.push("Redistribute: Scatter ID")

            timer.push("Redistribute: Scatter ID: Allocate")
            recv_id = np.empty([recv_size, k], dtype=np.int32)
            timer.pop("Redistribute: Scatter ID: Allocate")

            timer.push("Redistribute: Scatter ID: Comm")
            comm.Scatterv([send_block_ids, ssizes, sstarts, MPI.INT], recv_id, root=r)
            timer.pop("Redistribute: Scatter ID: Comm")

            timer.push("Redistribute: Scatter ID: Copy")
            result_ids[offset:offset+recv_size, ...] = recv_id
            timer.pop("Redistribute: Scatter ID: Copy")

            timer.pop("Redistribute: Scatter ID")

            #communicate Dist block
            
            timer.push("Redistribute: Scatter Data")

            timer.push("Redistribute: Scatter Data: Allocate")
            recv_dist = np.empty([recv_size, k], dtype=np.float32)
            timer.pop("Redistribute: Scatter Data: Allocate")

            timer.push("Redistribute: Scatter Data: Comm")
            comm.Scatterv([send_block_dist, ssizes, sstarts, MPI.FLOAT], recv_dist, root=r)
            timer.pop("Redistribute: Scatter Data: Comm")

            timer.push("Redistribute: Scatter Data: Copy")
            result_dist[offset:offset+recv_size, ...] = recv_dist
            timer.pop("Redistribute: Scatter Data: Copy")

            timer.pop("Redistribute: Scatter Data")

            offset += recv_size

        timer.push("Redistribute: Sort")
        lids = np.argsort(result_gids)
        new_gids = result_gids[lids]

        result_ids = result_ids[lids, ...]
        result_dist = result_dist[lids, ...]
        timer.pop("Redistribute: Sort")

        return new_gids, (result_ids, result_dist)

    class Node:

        verbose = False

        def __init__(self, tree, idx=0, level=0, size=0, gids=None):
            """Initalize a member of the RKDT.Node class

            Arguments:
                tree -- the owning RKDT (used for grabbing data from the pointset)

            Keyword Arguments:
                idx -- the binary tree array order index
                level -- the level in the tree of the node (root = level 0)
                size -- the number of points that this node corresponds to
                gids -- the list of global indicies for the owned points
            """

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
            #print(self.tree.offset_list[self.level])
            #print("idx", idx - 2**self.level +1)
            #self.offset = self.tree.offset_list[self.level][0]

            #print(idx, "Node Creation")
            #print(idx, "offset_list", self.tree.offset_list[self.level+1])
            #print(idx, "id", idx - 2**self.level + 1)
            #print(idx, 'level', self.level)
            self.offset = int(self.tree.offset_list[self.level+1][idx - 2**self.level + 1])
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
            """Return the binary tree array order index of a RKDT.Node """
            return self.id

        def get_gids(self):
            """Return the global indicies of the owned points"""
            return self.gids

        def get_reference(self):
            """Return the size x d array of the owned pointset. Note this produces a copy."""
            #self.tree.ordered=False;
            if self.tree.ordered:
                return self.tree.data[self.offset:self.offset+self.size, ...]
            else:
                return self.tree.data[self.gids, ...]

        def set_right_child(self, node):
            """Set the 'pointer' to the right child. (Points should be < split)"""
            self.children[0] = node

        def set_left_child(self, node):
            """Set the 'pointer' to the left child. (Points should be > split)"""
            self.children[1] = node

        def set_children(self, nodelist):
            """Set both children from nodelist = [left, right]. Update leaf status."""
            self.children = nodelist
            self.isleaf = True if all([child is None for child in nodelist]) else False

        def set_parent(self, node):
            """Set the 'pointer' to the parent node"""
            self.parent = node

        def average(self, idx=0):
            """Return the average of points in the node along a specified axis (0 < idx < d-1)

            Keyword Arguments:
                idx -- axis to compute average along. If idx < 0 compute with the saved local projection onto anchor points
            """
            return self.lib.mean(self.tree.host_data[self.gids, idx])

        def median(self, idx=0):
            """Return the median of points in the node along a specified axis (0 < idx < d-1)

            Keyword Arguments:
                idx -- axis to compute median along. If idx < 0 compute with the saved local projection onto anchor points
            """
            return self.lib.median(self.tree.host_data[self.gids, idx])

        def select_hyperplane(self):
            a = self.level
            if a >= self.tree.dim:
                a = random.randint(0, self.tree.dim-1)
            self.vector = self.tree.vectors[a, :]
            #print("vector", self.vector.shape)
            #print("data", self.tree.data[self.offset:self.offset+self.size, ...].shape)
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
            if self.tree.location == "CPU":

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

            elif self.tree.location == "GPU":

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
            #print("R", R)
            #print("self.gids aknn", self.gids)
            #print("self.tree.gids aknn", self.tree.gids)
            lids = np.arange(self.size, dtype=np.int32)
            results =  Primitives.single_knn(self.gids, R, Q, k)
            #print("Same as merge", results)
            #l, d = results

            #invp = np.argsort(self.gids)
            #invp = np.array(invp, dtype=np.int32)

            #results = (self.gids[l[:, :]], d[:, :])
            results = Primitives.merge_neighbors(results, results, k)
            return results
            

        def knn_all(self, k):
            """
            Perform an exact exhaustive all-knn search in the node. O(size x gids x d)

            Arguments:
                k -- number of nearest neighbors (Limitation: k < leafsize)
            """
            R = self.get_reference()
            return Primitives.single_knn(self.gids, R, R, k)

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

            #if self.tree.location == "CPU":
            #    dist = Primitives.distance(q, self.tree.data[self.anchors, ...])
            #    dist = dist[0] - dist[1]
            #else:
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


        #TODO: Make bookkeeping parallel

        for i in range(iters):
            start = batchsize*(i-1)
            stop  = batchsize*(i) if i < iters-1 else n_leaves #handle edgecase of last batch
 
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

        #rgL = []
        #for leaf in leaf_nodes:
        #    rgL.append(self.real_gids[leaf.gids])
        #real_gids = self.lib.concatenate(rgL)
        real_gids = self.real_gids

        #compute batchsize
        MAXBATCH = 2**28
        n_leaves = len(leaf_nodes)
        batchsize = n_leaves if n_leaves < MAXBATCH else MAXBATCH 
       
        #Loop over all batches 
        iters = int(np.ceil(n_leaves/batchsize))

        #TODO: Make bookkeeping parallel
        #print("STARTING BATCH:", iters)
        for i in range(iters):

            setup_t = time.time()
            batch_t = time.time()
            start = batchsize*(i)
            stop  = batchsize*(i+1) if i < iters-1 else n_leaves #handle edgecase of last batch
 
            gidsList = []
            RList = []
            #print(start, stop)
            #print(leaf_nodes[start:stop])
            #setup leaf lists
            for leaf in leaf_nodes[start:stop]:
                gidsList.append(leaf.gids)
                #gidsList.append(np.arange(leaf.size, dtype=np.int32))
                RList.append(leaf.get_reference())
                #print(leaf.get_reference().shape)
            

            #print(rank, "gidsList", gidsList)
            #print(rank, "gids", self.gids)
            #print(rank, "rgids", real_gids)
            
            #print("Populated RList")
            #call batch routine

            #print(gidsList)
            #print("RList", RList[0].shape)
            #print(k)
            setup_t = time.time() - setup_t
            if rank == 0:
                print("Search setup time took ", setup_t)

            timer.push("AKNN: Compute")
            comp_t = time.time()
            NLL, NDL, out = Primitives.multileaf_knn(gidsList, RList, RList, k)
            comp_t = time.time() - comp_t
            timer.pop("AKNN: Compute")

            if rank ==0:
                print("Search computation took ", comp_t)

            #print("Finished kernel call")
            timer.push("AKNN: Copy")
            copy_t = time.time()
            #populate results from temporary local objects
            j = 0;
            #print("out", out)
            for leaf in leaf_nodes[start:stop]:
                idx = leaf.get_gids()
                lk = min(k, leaf.size-1)
                NL =  NLL[j]
                ND =  NDL[j]
                #print(NL.shape)
                #print(NL)
                neighbor_list[idx, :] = real_gids[NL[:, :]]
                neighbor_dist[idx, :] = ND[:, :]
                del NL
                del ND
                j += 1

            timer.pop("AKNN: Copy")
            del NLL
            del NDL
            del out
            del gidsList
            del RList

            gc.collect()


            #for i in range(self.local_size):
            #    for j in range(k):
            #        neighbor_list[i,j] = real_gids[neighbor_list[i, j]]

            copy_t = time.time() - copy_t
            if rank ==0:
                print("Search copy took ", copy_t)

            #print("Finished copy")
            batch_t = time.time() - batch_t
            if rank ==0:
                print("Search took:", batch_t)

        timer.pop("AKNN")

        return neighbor_list, neighbor_dist

    def dist_exact(self, Q, k):
        size = self.comm.Get_size()
        rank = self.comm.Get_rank()

        query_size = Q.shape[0]
        root = self.nodelist[0]
        result =  root.knn(Q, k)
        #print(rank, "result", result)
        #print(rank, "datapoints", self.data[result[0][0, :]])
        recvbuff_list = np.empty([size, query_size, k], dtype=np.int32)
        recvbuff_dist = np.empty([size, query_size, k], dtype=np.float32)

        #recvbuff_rids = np.empty([size, self.local_size], dtype=np.int32)

        rgs = self.real_gids
        #print(rank, "rgs", rgs)

        ids = rgs[result[0]]
        #print(rank, "rids reordered", rank, ids)

        #rgs = rgs[self.gids]
        #print(rgs)
        #ids = rgs[result[0]]
        #print(rank, ids)

        #ids = result[0]

        #print(rank, "base", ids)

        self.comm.Gather(ids, recvbuff_list, root=0)
        self.comm.Gather(result[1], recvbuff_dist, root=0)
        #self.comm.Gather(rgs, recvbuff_rids, root=0)

        result = None

        if rank ==0:
            #print(rank, "# unique realgids", np.unique(np.concatenate(recvbuff_rids)))
            #print('len', len(recvbuff_list))
            #print("recvlist", recvbuff_list)
            #print("recvdist", recvbuff_dist)
            for i in range(size):
                neighbors = (recvbuff_list[i], recvbuff_dist[i])
                if result:
                    result = Primitives.merge_neighbors(result, neighbors, k)
                else:
                    result = neighbors
            #print("merged", result)
        return result
        
