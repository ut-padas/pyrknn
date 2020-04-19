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
    #print("local_size", n)

    #loop over gids, specify location
    for i in prange(n):
        locations[i] = int(gids[i]/n)
        #print(locations[i], gids[i], n)
    reorder_ids = np.argsort(locations, kind='mergesort')

    locations = locations[reorder_ids]
    lids = lids[reorder_ids]

    #print("reordered locations", locations)
    """
    new_locations = np.empty(n, dtype=np.int64)
    new_lids = np.empty(n, dtype=np.int64)
    for i in prange(n):
        idx = reorder_ids[i]
        new_locations[idx] = locations[i]
        new_lids[idx] = lids[i]

    del locations
    del lids
    locations = new_locations
    lids = new_lids
    print("my reordered locations", locations)
    print(lids)
    """
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
            #print(rank, "med", median, flush=True)
            #print(rank, "split", local_split=True)

            

            self.host_real_gids = self.host_real_gids[lids] 
            self.host_data = self.host_data[lids, ...]

            #self.host_data = self.reorder(lids, self.host_data)

            #print(rank, "original_proj", proj_data)
            #print(rank, "max_o_p", np.max(proj_data))
            #print(rank, "min_o_p", np.min(proj_data))
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



    def dist_build(self):

        #dist_t = time.time()
        comm = self.comm
        rank = comm.Get_rank()
        
        self.dist_levels = int(np.floor(np.log2(self.comm.Get_size())))
        self.levels = int(min( np.ceil(np.log2(np.ceil(self.size/self.leafsize))), self.levels ))
        if self.levels < self.dist_levels:
            self.levels = self.dist_levels
        #print("dist", self.dist_levels, flush=True)
        #print("lvls", self.levels, flush=True)
        

        #Create and share projection vectors
        if rank == 0:
            #print("lvls", self.levels) 
            vectors = np.random.rand(self.dim, self.levels)
            vectors = np.array(vectors, dtype=np.float32)

            qr_t = time.time()
            vectors = np.linalg.qr(vectors)[0]
            qr_t = time.time() - qr_t

            print("QR Time", qr_t, flush=True)
            #print(rank, vectors.shape)

            if self.dist_levels > self.dim:
                spill = np.random.randint(low=0, high=self.dim, size=self.dist_levels-self.dim)

        else:
            ld = min(self.dim, self.levels)
            vectors = np.zeros( (self.dim, ld), dtype=np.float32)
            if self.dist_levels > self.dim:
                spill = np.zeros(self.dist_levels-self.dim, dtype=np.int64)

        self.comm.Bcast(vectors, root=0)
        if self.dist_levels > self.dim:
            self.comm.Bcast(spill, root=0)

        vectors = vectors.T

        self.vectors = vectors

        #if self.location == "GPU":
        #    #Move to GPU
        #    self.vectors = self.lib.array(self.vectors, dtype=np.float32)

        #print("vec shape", self.vectors.shape, flush=True)

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
            #print(rank, "med", median, flush=True)

            self.host_real_gids = self.host_real_gids[lids] 
            #self.host_data = self.host_data[lids, ...]

            self.host_data = self.reorder(lids, self.host_data)

            #print(rank, "split", local_split)
            #print(rank, "original_proj", proj_data)
            #print(rank, "max_o_p", np.max(proj_data))
            #print(rank, "min_o_p", np.min(proj_data))
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

            MAX_SIZE= 2**24

            blocksize = int( MAX_SIZE / self.dim)

            #print(rank, "bs", blocksize, flush=True)

            #Grab views of what will be for redistributed
            sending_ids = self.host_real_gids[send_offset:send_offset+send_size]

            if self.sparse:
                rows = self.host_data.row
                cols = self.host_data.col
                d    = self.host_data.data
                sending_row = rows[send_offset:send_offset+send_size]
                sending_col = cols[send_offset:send_offset+send_size]
                sending_data = d[send_offset:send_offset+send_size]
            else:
                sending_data = self.host_data[send_offset:send_offset+send_size, ...]

            #print(rank, "median", median, flush=True)
            #print(rank, "max_proj_before", np.max(sending_data @ vector), flush=True)
            #print(rank, "min_pro_before", np.min(sending_data @ vector), flush=True)

            #Loop over nonblocking sends
            send_id_requests = deque()
            if self.sparse:
                send_data_requests = deque()
                send_row_requests = deque()
                send_col_requests = deque()
            else:
                send_data_requests = deque()

            offset = 0
            for message in send_dict[rank]:
                message_size = message[1]
                target = message[0]
                tag = message[2]

                #Create local views
                send_id = sending_ids[offset:offset+message_size]
                send_data = sending_data[offset:offset+message_size, ...]

                if self.sparse:
                    send_row = sending_row[offset:offset+message_size]
                    send_col = sending_col[offset:offset+message_size]


                blocksize = int( MAX_SIZE / self.dim)
                nblocks = message_size // blocksize
                iters = int(np.ceil(message_size / blocksize))
                #print(rank, message_size, 'nblocks', iters, flush=True)
                #print(rank, "Creating send: ", rank, " -> ", target, " tag: ", tag, "size:", 4*self.dim*message_size, "size:", self.host_gids.itemsize*message_size, flush=True)

                block_offset = 0
                for j in range(iters):
                    if j == iters-1:
                        blocksize = message_size % blocksize 
                                            
                    t = j + 1

                    #Send IDS

                    #print(rank, "Creating send: ", rank, " -> ", target, " tag: ", t*mpi_size+tag+1, "size:", 4*self.dim*message_size, "size:", self.host_gids.itemsize*message_size, j,flush=True)
                    send_id_block = send_id[block_offset:block_offset+blocksize]
                    temp_id_req = comm.isend(send_id_block, dest=target, tag=t*mpi_size+tag+1)
                    send_id_requests.append(temp_id_req)
                    
                    #Send Data
                    send_data_block = send_data[block_offset:block_offset+blocksize, ...]
                    #print(rank, "shape of sent_data", send_data_block.shape, flush=True)
                    if self.sparse:
                        send_row_block = send_row[block_offset:block_offset+blocksize]
                        send_row_block = send_row_block - send_row_block[0]

                        send_col_block = send_col[block_offset:block_offset+blocksize]

                        #print(rank,"->", target, "dat", mpi_size*(nblocks+1) + t*mpi_size + mpi_size+tag+1, flush=True)
                        temp_data_req = comm.isend(send_data_block, dest=target, tag=mpi_size*(nblocks+1) + t*mpi_size + mpi_size+tag+1)
                        send_data_requests.append(temp_data_req)

                        #print(rank, "->",  target, "col", mpi_size*(nblocks+2) + t*mpi_size + mpi_size+tag+1, flush=True)
                        temp_col_req = comm.isend(send_col_block, dest=target, tag=mpi_size*(nblocks+2) + t*mpi_size+ mpi_size +tag+1)
                        send_col_requests.append(temp_col_req)

                        #print(rank,"->", target, "row", mpi_size*(nblocks+3) + t*mpi_size + mpi_size+tag+1, flush=True)
                        temp_row_req = comm.isend(send_row_block, dest=target, tag=mpi_size*(nblocks+3) + t*mpi_size+ mpi_size + tag+1)
                        send_row_requests.append(temp_row_req)

                    else:
                        #print(rank, "send", mpi_size, nblocks, t, tag, flush=True)
                        #print(rank, "Creating send: ", rank, " -> ", target, " tag: ", mpi_size*(nblocks+1) + t*mpi_size+tag+1, "size:", self.host_data.itemsize*self.dim*blocksize, "size:", self.host_gids.itemsize*message_size, flush=True)
                        temp_data_req = comm.isend(send_data_block, dest=target, tag=mpi_size*(nblocks+1) + t*mpi_size + tag+1)
                        send_data_requests.append(temp_data_req)

                    block_offset += blocksize
                offset += message_size

            #print(rank, "Finished queueing sends")

            blocksize = int( MAX_SIZE / self.dim)

            recv_id_requests = deque()
            if self.sparse:
                recv_data_requests = deque()
                recv_row_requests = deque()
                recv_col_requests = deque()
            else:
                recv_data_requests = deque()

            for message in recv_dict[rank]:

                message_size = message[1]
                source = message[0]
                tag = message[2]

                blocksize = int( MAX_SIZE / self.dim)
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
                        #print(rank, "<-", source, "rec dat", mpi_size*(nblocks+1) + t*mpi_size + mpi_size+tag+1, flush=True)
                        temp_data_req = comm.irecv(10*self.host_data.data.itemsize*blocksize + 10, source=source, tag=mpi_size*(nblocks+1) + t*mpi_size + mpi_size + tag + 1)
                        recv_data_requests.append(temp_data_req)

                        #print(rank, "<-", source, "rec col", mpi_size*(nblocks+2) + t*mpi_size + mpi_size+tag+1, flush=True)
                        temp_col_req = comm.irecv(10*self.host_data.col.itemsize*blocksize + 10, source=source, tag=mpi_size*(nblocks+2) + t*mpi_size + mpi_size + tag + 1)
                        recv_col_requests.append(temp_col_req)

                        #print(rank, "<-", source, "rec row", mpi_size*(nblocks+3) + t*mpi_size + mpi_size+tag+1, flush=True)
                        temp_row_req = comm.irecv(10*self.host_data.row.itemsize*blocksize + 10, source=source, tag=mpi_size*(nblocks+3) + t*mpi_size + mpi_size + tag + 1)
                        recv_row_requests.append(temp_row_req)
                    else:
                        #print(rank, "recv", mpi_size, nblocks, t, tag, flush=True)
                        #print(rank, "Creating recv: ", rank, " <- ", source, " tag: ", mpi_size*(nblocks+1) + t*mpi_size+tag+1, "size:", self.host_data.itemsize*self.dim*message_size, "size:", self.host_gids.itemsize*message_size, flush=True)
                        temp_data_req = comm.irecv(10*self.host_data.itemsize*self.dim*blocksize + 10, source=source, tag=mpi_size*(nblocks+1) + t*mpi_size+tag+1)
                        recv_data_requests.append(temp_data_req)

            #Alternate sends and recvs from high/low

            #Get maximum number of sends
            local_sends = len(send_id_requests)
            local_sends = np.array(local_sends, dtype=np.int32)
            #print(rank, "send_id", len(send_id_requests), flush=True)
            max_sends = np.array(0, dtype=np.int32)
            comm.Allreduce(local_sends, max_sends, op=MPI.MAX)
            #print(rank, "max sends", max_sends, flush=True)
            #print(rank, "data_sends", len(send_data_requests), flush=True)
            #print(rank, "col_sends", len(send_col_requests), flush=True)
            #print(rank, "row_sends", len(send_row_requests), flush=True)
            #print(rank, "data_Recvs", len(recv_data_requests), flush=True)
            #print(rank, "col_recvs", len(recv_col_requests), flush=True)
            #print(rank, "row_recvs", len(recv_row_requests), flush=True)
            #print(rank, "id_recvs", len(recv_id_requests), flush=True)


            id_offset = 0
            data_offset = 0
            if self.sparse:
                row_offset = 0
                col_offset = 0

            for i in range(2*max_sends):
                active = False
                if i%2 == 0:
                    if rank >= half:
                        active = True
                else:
                    if rank < half:
                        active = True

                #if active:
                #Send data to other half
                #print(rank, "sending", flush=True)
                if len(send_id_requests) > 0:
                    #print(rank, "s wait id", flush=True) 
                    req = send_id_requests.popleft()
                    req.wait()
                    #print(rank, "sent ids", flush=True)

                if len(send_data_requests) > 0:
                    #print(rank, "s wait dat", flush=True)
                    req = send_data_requests.popleft()
                    req.wait()
                    #print(rank, "sent data", flush=True)

                if self.sparse:
                    #print(rank, "is sparse", flush=True)
                    if len(send_col_requests) > 0:
                        #print(rank, "s wait col", flush=True)
                        req = send_col_requests.popleft()
                        req.wait()
                        #print(rank, "sent cols", flush=True)

                    if len(send_row_requests) > 0:
                        #print(rank, "s wait row", flush=True)
                        req = send_row_requests.popleft()
                        req.wait()
                        #print(rank, "sent rows", flush=True)

                #print(rank, "finished sending all", flush=True)
#                else:
                #Recv data from other half
                #print(rank, "recving", flush=True)
                if len(recv_id_requests) > 0:
                    #print(rank, "wait id", flush=True)
                    req = recv_id_requests.popleft()
                    new_id = req.wait()
                    message_size = new_id.shape[0]
                    sending_ids[id_offset:id_offset+message_size] = new_id
                    id_offset += message_size
                    #print(rank, "recv id", flush=True)
                
                if len(recv_data_requests) > 0:
                    #print(rank, "wait data", flush=True)
                    req = recv_data_requests.popleft()
                    new_data = req.wait()
                    message_size = new_data.shape[0]
                    #print(rank, "New_data-shape", new_data.shape, flush=True)
                    sending_data[data_offset:data_offset+message_size, ...] = new_data
                    data_offset += message_size
                    #print(rank, "recved data", flush=True)

                #print(rank, "here", flush=True)
                if self.sparse:
                    if len(recv_col_requests) > 0:
                        #print(rank, "wait col", flush=True)
                        req = recv_col_requests.popleft()
                        new_col = req.wait()
                        sending_col[col_offset:col_offset+message_size] = new_col
                        col_offset += message_size
                        #print(rank, "recved col", flush=True)
                    if len(recv_row_requests) > 0:
                        #print(rank, "wait row", flush=True)
                        req = recv_row_requests.popleft()
                        new_row = req.wait()
                        sending_row[row_offset:row_offset+message_size] = new_row + row_offset + send_offset
                        row_offset += message_size
                        #print(rank, "recved row", flush=True)

            comm.Barrier()

            if self.sparse: 
                self.host_data = sp.coo_matrix((d, (rows, cols)))    
            else:
                self.host_data[send_offset:send_offset+send_size] = sending_data

            self.host_real_gids[send_offset:send_offset+send_size] = sending_ids

            #print(rank, "median", median, flush=True)
            #print(rank, "sd size", sending_data.shape, flush=True)
            #print(rank, "max_proj", np.max(sending_data @ vector), flush=True)
            #print(rank, "min_proj", np.min(sending_data @ vector), flush=True)
            
            median_list.append(median)           

            #Split communicator
            comm = comm.Split(color, rank)

        #dist_t = time.time() - dist_t

        rank = self.comm.Get_rank()
        #print("Distributed Build Time:", dist_t)
        #print("build dist copy_t", copy_t_f)
        #print("build dist collect_t", collect_t_f)
        #print(rank, "real gids", self.host_real_gids)
        if self.sparse:
            temp = self.host_data.tocsr()
            data = temp.data
            ind  = temp.indices
            ptr  = temp.indptr
            
            #Convert to correct datatype and location
            data = np.array(data, dtype=np.float32)
            ind  = np.array(ind, dtype=np.int32)
            ptr  = np.array(ptr, dtype=np.int32)

            self.data = sp.csr_matrix((data, ind, ptr))
        else:
            self.data = np.array(self.host_data)

        self.gids = np.array(self.host_gids, dtype=np.int32)
        self.real_gids = np.array(self.host_real_gids, dtype=np.int32)


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


        rank = self.comm.Get_rank()
        #Find out the maximum number of levels required
        #TODO: Fix definition of leafsize to make this proper. Can often overestimate level by one.
        self.levels = int(min( np.ceil(np.log2(np.ceil(self.size/self.leafsize))), self.levels ))
        self.dist_levels = int(np.floor(np.log2(self.comm.Get_size())))
        #print(self.levels)
        #print(self.dist_levels)
        
        dist_t = time.time()

        self.dist_build()

        dist_t = time.time() - dist_t
        #print(rank, "dist_t", dist_t)
 
        #print(self.vectors)
    

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

        #print(self.offset_list)

        self.nodelist = [None] * (2**(self.levels - self.dist_levels + 1)-1)
        #print("local levels:", len(self.nodelist), flush=True)
        
        #Copy over from host data to data
        """        
        #Distributed Level by Level construction
        #GOFMM Style (only support p = 2^x)
        comm = self.comm
        median_list = []

        collect_t_f = 0
        wait_t_f = 0
        copy_t_f = 0

        for i in range(self.dist_levels):
            #Project and Reorder
            rank = comm.Get_rank()
            mpi_size = comm.Get_size()

            collect_t = time.time()

            size = np.array(0, dtype=np.int32)
            local_size = np.array(self.local_size, dtype=np.int32)
            comm.Allreduce(local_size, size, op=MPI.SUM)

            a = i
            if i >= self.dim:
                a = random.randint(0, self.tree.dim-1)
            vector = self.dist_vectors[a, :]
            #print(rank, vector.shape)

            proj_data = self.host_data @ vector

            lids = np.arange(self.local_size)
            lids = np.array(lids, dtype=np.int32)
            print(rank, size/2, proj_data.shape, lids.shape, flush=True)
            median, local_split = Primitives.dist_select(size/2, proj_data, lids, comm)
            print(rank, "med", median, flush=True)
            self.host_real_gids = self.host_real_gids[lids] 
            self.host_data = self.host_data[lids, ...]

            #print(rank, "original_proj", proj_data)
            #print(rank, "max_o_p", np.max(proj_data))
            #print(rank, "min_o_p", np.min(proj_data))
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
           
            #An unfortunatly complex load balancing computation
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
                #print(rank, "swapping send/recv", flush=True)
                temp = send_dict 
                send_dict = recv_dict
                #recv_dict = temp
            else:
                recv_dict = send_dict

            print(rank, "Send Dict", send_dict, flush=True)
            print(rank, "Recv Dict", recv_dict, flush=True)
            collect_t_f += time.time() - collect_t

            #Grab memory for redistribute
            sending_ids = self.host_real_gids[send_offset:send_offset+send_size]
            sending_data = self.host_data[send_offset:send_offset+send_size]

            #print(rank, "median", median, flush=True)
            #print(rank, "max_proj_before", np.max(sending_data @ vector), flush=True)
            #print(rank, "min_pro_before", np.min(sending_data @ vector), flush=True)


            #Loop over nonblocking sends
            send_id_requests = []
            send_data_requests = []
            offset = 0
            for message in send_dict[rank]:
                message_size = message[1]
                target = message[0]
                tag = message[2]
                print(rank, "Creating send: ", rank, " -> ", target, " tag: ", tag, "size:", self.host_data.itemsize*self.dim*message_size, "size:", self.host_gids.itemsize*message_size, flush=True)
                #Send IDS
                send_id = sending_ids[offset:offset+message_size]
                temp_id_req = comm.isend(send_id, dest=target, tag=tag+1)
                send_id_requests.append(temp_id_req)
                
                #Send Data
                send_data = sending_data[offset:offset+message_size]
                temp_data_req = comm.isend(send_data, dest=target, tag=100*mpi_size*(tag+1))
                send_data_requests.append(temp_data_req)

                offset += message_size

            #print(rank, "Finished queueing sends")

            recv_id_requests = []
            recv_data_requests = []
            for message in recv_dict[rank]:
                message_size = message[1]
                source = message[0]
                tag = message[2]
                
                print(rank, "Creating recv: ", rank, " <- ", source, " tag: ", tag, "size:", self.host_data.itemsize*self.dim*message_size, "size:", self.host_gids.itemsize*message_size, flush=True)
                #Recv IDS
                temp_id_req = comm.irecv(10*self.host_gids.itemsize*message_size + 10, source=source, tag=tag+1)
                recv_id_requests.append(temp_id_req)
                                   
                #Recv Data
                temp_data_req = comm.irecv(10*self.host_data.itemsize*self.dim*message_size + 10, source=source, tag=100*mpi_size*(tag+1))
                recv_data_requests.append(temp_data_req)

            print("wait id")

            for req in send_id_requests:
                req.wait()

            print("wait recv id")

            copy_t = time.time()

            offset = 0
            for req in recv_id_requests:
                new_id = req.wait()
                message_size = new_id.shape[0]
                sending_ids[offset:offset+message_size] = new_id
                offset += message_size


            print("wait data")
            for req in send_data_requests:
                req.wait()

            print("wait recv data")
            offset = 0
            for req in recv_data_requests:
                new_data = req.wait()
                message_size = new_data.shape[0]
                sending_data[offset:offset+message_size] = new_data

                #print(rank, "median", median, flush=True)
                #print(rank, "max_proj_inc", np.max(new_data @ vector), flush=True)
                #print(rank, "min_pro_incj", np.min(new_data @ vector), flush=True)

                offset += message_size

            copy_t_f += time.time() - copy_t

            comm.Barrier()

            #print(rank, "vector", vector, flush=True)

            self.host_real_gids[send_offset:send_offset+send_size] = sending_ids
            self.host_data[send_offset:send_offset+send_size] = sending_data

            #print(rank, "median", median, flush=True)
            #print(rank, "max_proj", np.max(sending_data @ vector), flush=True)
            #print(rank, "min_proj", np.min(sending_data @ vector), flush=True)
            
            median_list.append(median)           

            #Split communicator
            comm = comm.Split(color, rank)


        dist_t = time.time() - dist_t
        print("Distributed Build Time:", dist_t)
        print("build dist copy_t", copy_t_f)
        print("build dist collect_t", collect_t_f)
        """

        self.data = self.lib.array(self.host_data, dtype=np.float32)
        self.gids = self.lib.array(self.host_gids, dtype=np.int32)
        self.real_gids = self.lib.array(self.host_real_gids, dtype=np.int32)

        if self.sparse:
            print("Removed as it need changes soon. Just call aknn_all directly.")
            self.built = False
            #self.ordered=True
        else:
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
                        #TODO: Key area for PARLA Tasks
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
                                    #print("A node in level", np.log(start+1), "took ", split_t) 
                        await T
                        self.built=True
            
            elif self.location == "GPU":
                root = self.Node(self, idx=0, level=0, size=self.local_size, gids=self.gids)
                self.root = root
                root.split(cp.cuda.Stream(non_blocking=True))
                self.built=True

            self.ordered = True
       

 
        #Fix overestimate of tree levels (see #TODO above)
        #print(self.nodelist)
        #print(len(self.nodelist))
        #print(self.levels-self.dist_levels)
        self.levels = self.levels-self.dist_levels
        while self.get_level(self.levels)[0] is None:
            self.levels -= 1
            self.nodelist = self.nodelist[:2**(self.levels+1)-1]


        #print("FINAL DATA", self.data)
        #print("FINAL GIDS", self.gids)

    def redist(self, neighbors):
        neighbor_ids = neighbors[0]
        neighbor_dist = neighbors[1]        

        k = neighbor_ids.shape[1]

        mpi_size = self.comm.Get_size()
        rank = self.comm.Get_rank()
        comm = self.comm

        result_gids = np.zeros(self.local_size, dtype=np.int32)
        result_ids  = np.zeros((self.local_size, k), dtype=np.int32)
        result_dist = np.zeros((self.local_size, k), dtype=np.float32)

        if self.location == "GPU":
            real_gids = self.host_real_gids
        else:
            real_gids = self.real_gids

        lids = np.arange(self.local_size, dtype=np.int32)

        collect_t = time.time()
        
        starts, lengths, lids = collect(lids, real_gids, mpi_size)

        all_lengths = []
        for i in range(mpi_size):
            if i == rank:
                data = lengths
            else:
                data = None
            data = self.comm.bcast(data, root=i)
            all_lengths.append(data)

        collect_t = time.time() - collect_t

        sending_ids = neighbor_ids[lids, ...]
        sending_dist = neighbor_dist[lids, ...]
        sending_gids = real_gids[lids]

        offset = 0
        for r in range(mpi_size):

            split_sizes = all_lengths[r]
            split_starts = starts

            #print(rank, "sizes", split_sizes)
            #print(rank, "starts", split_starts)

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
            recv_gid = np.empty(recv_size, dtype=np.int32)
            comm.Scatterv([send_block_gids, ssizes, sstarts, MPI.INT], recv_gid, root=r)
            result_gids[offset:offset+recv_size, ...] = recv_gid
            #print(rank, "sendgid", sending_gids)
            #print(rank, recv_gid)

            split_sizes = split_sizes * k
            split_starts = starts * k

            if r == rank:
                sstarts = tuple(split_starts)
                ssizes  = tuple(split_sizes)

            else:
                ssizes = None
                sstarts = None

            #communicate ID block
            recv_id = np.empty([recv_size, k], dtype=np.int32)
            comm.Scatterv([send_block_ids, ssizes, sstarts, MPI.INT], recv_id, root=r)
            result_ids[offset:offset+recv_size, ...] = recv_id

            #print(rank, recv_id)

            #communicate Dist block
            
            recv_dist = np.empty([recv_size, k], dtype=np.float32)
            comm.Scatterv([send_block_dist, ssizes, sstarts, MPI.FLOAT], recv_dist, root=r)
            result_dist[offset:offset+recv_size, ...] = recv_dist

            offset += recv_size

        lids = np.argsort(result_gids)
        new_gids = result_gids[lids]

        #print(rank, "sorted_new_gids", new_gids)
        result_ids = result_ids[lids, ...]
        result_dist = result_dist[lids, ...]

        #print(rank, "end redist", flush=True)
        return new_gids, (result_ids, result_dist)


    def redistribute(self, neighbors):
        neighbor_ids = neighbors[0]
        neighbor_dist = neighbors[1]        

        k = neighbor_ids.shape[1]

        mpi_size = self.comm.Get_size()
        rank = self.comm.Get_rank()

        new_gids = np.zeros(self.local_size, dtype=np.int32)
        result_ids  = np.zeros((self.local_size, k), dtype=np.int32)
        result_dist = np.zeros((self.local_size, k), dtype=np.float32)

        if self.location == "GPU":
            real_gids = self.host_real_gids
        else:
            real_gids = self.real_gids

        #Collect gids in order
        #Redistributed Order
        #lids = self.host_gids
        #Reorganized order
        lids = np.arange(self.local_size, dtype=np.int32)

        collect_t = time.time()
        #if self.location == "CPU":
        #    real_gids = real_gids[self.gids]

        #print(rank, "gids", real_gids, flush=True)
        #print(rank, "gids_max", np.max(gids), flush=True)
        #print(rank, "gids_dev", np.max(gids)/self.local_size, flush=True)

        #bins = defaultdict(list)
        #for i in range(len(real_gids)):
        #    bins[real_gids[i]//self.local_size].append(i)
        
        starts, lengths, lids = collect(lids, real_gids, mpi_size)

        #print(rank, "bins", bins, flush=True)
        #print(rank, "starts", starts, flush=True)
        #print(rank, "to 0", lids[starts[0]:starts[0]+lengths[0]], flush=True)
        #print(rank, "to 1", lids[starts[1]:starts[1]+lengths[1]], flush=True)
        #print(rank, "to 2", lids[starts[2]:starts[2]+lengths[2]], flush=True)
        #print(rank, "to 3", lids[starts[3]:starts[3]+lengths[3]], flush=True)

        #Get length of each local rank ownership
        #lengths = []
        #for i in range(mpi_size):
        #    lengths.append(len(bins[i]))        

        all_lengths = []
        for i in range(mpi_size):
            if i == rank:
                data = lengths
            else:
                data = None
            data = self.comm.bcast(data, root=i)
            all_lengths.append(data)

        collect_t = time.time() - collect_t
        #print("collect_t", collect_t)


        #print(rank, "all_lengths", all_lengths, flush=True)

        startup_t = time.time()

        comm = self.comm

        MAX_SIZE = 2**24

        send_reqs_id   = deque()
        send_reqs_data_ids = deque()
        send_reqs_data_dist = deque()

        recv_reqs_id   = deque()
        recv_reqs_data_ids = deque()
        recv_reqs_data_dist = deque()

        for i in range(mpi_size):

            i = (rank+i)%mpi_size
            if i != rank and lengths[i] > 0:

                message_size = lengths[i]
                blocksize = int( MAX_SIZE / k)
                nblocks = message_size // blocksize
                iters = int(np.ceil(message_size / blocksize))

                sending_gids = lids[starts[i]:starts[i]+lengths[i]]
                #sending_gids = np.array(bins[i], dtype=np.int32)
                sending_data_ids = neighbor_ids[sending_gids, ...]
                sending_data_dist = neighbor_dist[sending_gids, ...]
                sending_gids = real_gids[sending_gids]

                block_offset = 0
                for j in range(iters):

                    if j == iters-1:
                        blocksize = message_size % blocksize
 
                    sending_gids_block      = sending_gids[block_offset:block_offset+blocksize]
                    sending_data_ids_block  = sending_data_ids[block_offset:block_offset+blocksize, ...]
                    sending_data_dist_block = sending_data_dist[block_offset:block_offset+blocksize, ...]

                    #print(rank,"Creating send from", rank, "->", i, "tag", (iters+2)*mpi_size + (j+1)*mpi_size +i, flush=True)
                    #print(rank,"Creating send from", rank, "->", i, "tag", (iters+4)*mpi_size + (j+1)*mpi_size +i, flush=True)

                    send_req_id         = comm.isend(sending_gids, dest=i, tag=(iters+2)*mpi_size +(j+1)*mpi_size + i)
                    send_req_data_ids   = comm.isend(sending_data_ids, dest=i, tag=(iters+4)*mpi_size + (j+1)*mpi_size +i)
                    send_req_data_dist  = comm.isend(sending_data_dist, dest=i, tag=(iters+6)*mpi_size + (j+1)*mpi_size +i)

                    send_reqs_id.append(send_req_id)
                    send_reqs_data_ids.append(send_req_data_ids)
                    send_reqs_data_dist.append(send_req_data_dist)

            incoming_size = all_lengths[i][rank]
            if(i!=rank and incoming_size > 0):

                message_size = incoming_size
                blocksize = int( MAX_SIZE / k)
                nblocks = message_size // blocksize
                iters = int(np.ceil(message_size / blocksize))

                for j in range(iters):

                    if j == iters-1:
                        blocksize = message_size % blocksize 

                    #print(rank, "Creating recv from", rank ,"<-", i, "tag", (iters+2)*mpi_size+(j+1)*mpi_size+rank, flush=True)
                    #print(rank, "Creating recv from", rank ,"<-", i, "tag", (iters+4)*mpi_size+(j+1)*mpi_size+rank, flush=True)
                    #print(rank, "Incoming size should be", incoming_size, flush=True)
                    recv_req_id        = comm.irecv(10 * blocksize * new_gids.itemsize,    source=i, tag=(iters+2)*mpi_size+(j+1)*mpi_size+rank)
                    recv_req_data_ids  = comm.irecv(10 * blocksize * result_ids.itemsize * k,source=i, tag=(iters+4)*mpi_size+(j+1)*mpi_size+rank)
                    recv_req_data_dist = comm.irecv(10 * blocksize * result_dist.itemsize * k, source=i, tag=(iters+6)*mpi_size+(j+1)*mpi_size+rank)

                    recv_reqs_id.append(recv_req_id)
                    recv_reqs_data_ids.append(recv_req_data_ids)
                    recv_reqs_data_dist.append(recv_req_data_dist)
                
        startup_t = time.time() - startup_t
        #print("startup_t", startup_t)

        max_send = max(len(recv_reqs_id), len(send_reqs_id))

        id_offset = 0
        list_offset = 0
        dist_offset = 0

        for i in range(2*max_send):
            if len(send_reqs_id) > 0:
                req = send_reqs_id.popleft()
                req.wait()

            if len(send_reqs_data_ids) >0:
                req = send_reqs_data_ids.popleft()
                req.wait()

            if len(send_reqs_data_dist) > 0:
                req = send_reqs_data_dist.popleft()
                req.wait()

        #print(rank, "finished sending all", flush=True)

        for i in range(2*max_send):
            if len(recv_reqs_id) > 0:
                req = recv_reqs_id.pop()
                new_ids = req.wait()
                message_size = new_ids.shape[0]
                new_gids[id_offset:id_offset+message_size] = new_ids
                id_offset += message_size

            if len(recv_reqs_data_ids) > 0:
                req = recv_reqs_data_ids.pop()
                new_data = req.wait()
                message_size = new_data.shape[0]
                result_ids[list_offset:list_offset+message_size, ...] = new_data
                list_offset+=message_size

            if len(recv_reqs_data_dist) > 0:
                req = recv_reqs_data_dist.pop()
                new_dist = req.wait()
                message_size = new_dist.shape[0]
                result_dist[dist_offset:dist_offset+message_size, ...] = new_dist
                dist_offset+=message_size
            #print(rank, "recv", i, len(recv_reqs_id), len(recv_reqs_data_ids), len(recv_reqs_data_dist), flush=True)
            
        comm.Barrier()
        #print(rank, "Begin self", flush=True)
        #copy self
        self_size = lengths[rank]

        #l_gids = np.array(bins[rank], dtype=np.int32)
        l_gids = lids[starts[rank]:starts[rank]+lengths[rank]]

        self_data_ids = neighbor_ids[l_gids, ...]
        self_data_dist = neighbor_dist[l_gids, ...]
        self_gids = real_gids[l_gids]

        new_gids[id_offset:id_offset+self_size] = self_gids
        result_ids[list_offset:list_offset+self_size, ...] = self_data_ids
        result_dist[dist_offset:dist_offset+self_size, ...] = self_data_dist

        #end copy self
        #print(rank, "end self", flush=True)

        #print(rank, "new_gids", new_gids)
        lids = np.argsort(new_gids)
        new_gids = new_gids[lids]

        #print(rank, "sorted_new_gids", new_gids)
        result_ids = result_ids[lids, ...]
        result_dist = result_dist[lids, ...]
        #print(rank, "end redist", flush=True)
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

            """Select the random line and project onto it.
                Algorithm:
                    Select 2 random points owned by this node.
                    Compute the distance from all other points to these two 'anchor points'
                    The projection local_ is their difference.

                    This computes <x, (a_1 - a_2)>

            """
            #TODO: Replace with gpu kernel
            #print("shape", self.tree.vectors.shape)
            #print(self.id, "size", self.size)
            #print(self.id, "off", self.offset)
            #print(self.id, "lvl", self.level)
            a = self.level+self.tree.dist_levels
            if a >= self.tree.dim:
                a = random.randint(0, self.tree.dim-1)
            #print(self.id, "a", self.level)
            self.vector = self.tree.vectors[a, :]
            #self.vector = self.lib.random.rand(self.tree.dim, 1).astype(np.float32)
            #self.vector = self.vector/self.lib.linalg.norm(self.vector, 2)
            #print("DAT", self.tree.data[self.gids, ...].shape, flush=True)
            #print("vec", self.vector.shape, flush=True)
            self.local_ = self.tree.data[self.offset:self.offset+self.size, ...] @ self.vector
            del a
            #print("out", self.local_.shape, flush=True)
            #self.local_ = self.local_[:, 0]
            #self.anchors = self.lib.random.choice(self.gids, 2, replace=False)
            #dist = Primitives.distance(self.tree.data[self.gids, ...], self.tree.data[self.anchors, ...])
            #self.local_ = dist[0] - dist[1]


        def cleanup(self):
            del self.local_

        def split(self, stream=None):

            """Split a node and assign both children.

            Return value:
                children -- [left, right] containing the left and right child nodes respectively

            Algorithm:
                Project onto line. (un-normalized in the current implementation)
                Split at median of projection.
                Partition gids and assign to children. left < median, right > median

            """

            middle = int(self.size//2)
            #print("NODE :", self.id, flush=True)
            self.tree.nodelist[self.id] = self

            #Stop the split if the leafsize is too small, or the maximum level has been reached

            if (middle < self.tree.leafsize) or (self.level+1) > self.tree.levels:
                self.plane = None
                self.anchors = None
                self.vector = None
                self.isleaf=True
                return [None, None]

            #self.offset = self.tree.offset_list[self.level][idx - 2**self.level + 1]
            self.ordered = True            
            if self.tree.location == "CPU":

                #project onto line (projection stored in self.local_)
                self.select_hyperplane()
              
                #print(self.id, "local_", self.local_.shape, flush=True)
                #print(self.id, "size", self.size, flush=True)

                self.lids = self.lib.argpartition(self.local_, middle)  #parition the local ids

                
                self.tree.gids[self.offset:self.offset+self.size] = self.gids[self.lids]                  #partition the global ids

                self.gids = self.tree.gids[self.offset:self.offset+self.size]
                #print(self.gids)
                #gids_view = self.gids[self.offset:self.offset+self.size]
                #print(self.id, "data", self.tree.data.shape, flush=True)
                #print(self.id, "offset", self.offset, flush=True)
                #print(self.id, "size", self.size, flush=True)
                #print(self.id, "local_", self.local_.shape, flush=True)
                #print(self.id, "mid", middle, flush=True)
                #print(self.id, "lid", self.lids.shape, flush=True)
                self.plane = self.local_[self.lids[middle]]       #save the splitting line

                #self.cleanup()                                    #delete the local projection (it isn't required any more)

                #Initialize left and right nodes
                
                left = self.tree.Node(self.tree, level = self.level+1, idx = 2*self.id+1, size=middle, gids=self.gids[:middle])
                right = self.tree.Node(self.tree, level = self.level+1, idx = 2*self.id+2, size=int(self.size - middle), gids=self.gids[middle:])

                left.set_parent(self)
                right.set_parent(self)

                children = [left, right]
                self.set_children(children)
                
                local_data = self.tree.data[self.offset:self.offset+self.size]
                self.tree.data[self.offset:self.offset+self.size] = local_data[self.lids]

                #print("-----")
                #print(self.id, "lids", self.lids)
                #print(self.id, "self.local", self.local_)
                #print(self.id, "self.local reordered", self.local_[self.lids])
                #print(self.id, "self.gids", self.gids)
                
                #print(self.id, "median", self.plane)
                #print(self.id, "data", self.tree.data)
                #print(self.id, "self.tree.gids", self.tree.gids)

                #print(self.id, "offset", self.offset)
                #print(self.id,"size", self.size)

                #print("-----")

                #del local_data
                #del self.lids
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
                    #del self.lids
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
            result = self.lib.asarray(self.data[self.gids, ...], dtype='float32')
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
            print("Search setup time took ", setup_t)

            comp_t = time.time()
            NLL, NDL, out = Primitives.multileaf_knn(gidsList, RList, RList, k)
            comp_t = time.time() - comp_t
            print("Search computation took ", comp_t)

            #print("Finished kernel call")

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
            print("Search copy took ", copy_t)

            #print("Finished copy")
            batch_t = time.time() - batch_t
            print("Search took:", batch_t)

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
        
