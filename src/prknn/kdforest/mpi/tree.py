from . import error as ErrorType
from . import util as Primitives

from parla import Parla
from parla.array import copy, storage_size
from parla.cpu import cpu
from parla.tasks import *

import time
import os

import numpy as np
import cupy as cp
import scipy.sparse as sp
import random
 
from collections import defaultdict
from mpi4py import MPI

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
            comm = MPI.COMM_WORLD
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

        if (pointset is not None):
            self.local_size = len(pointset)           #the number of points in the pointset
            
            self.gids = self.lib.arange(self.local_size, dtype=np.int32)    #the global ids of the points in the pointset (assign original ordering)

            if( self.sparse ):
                #(data, indices, indptr)

                if(self.location == "GPU"):
                    #Copy of data in CPU Memory

                    local_data = self.lib.asnumpy(pointset.data)
                    local_indices = self.lib.asnumpy(pointset.indices)
                    local_indptr = self.lib.asnumpy(pointset.indptr)

                    self.host_data = sp.csr_matrix( (local_data, local_indicies, local_indptr) )

                #Copy of data in location Memory
                local_data = self.lib.asarray(pointset.data, dtype=np.float32)
                local_indices = self.lib.asarray(pointset.indices, dtype=np.int32)
                local_indptr = self.lib.asarray(pointset.indptr, dtype=np.int32)
                self.data = self.lib.csr_matrix( (local_data, local_indicies, local_indptr) )

            else:
                if(self.location == "GPU"):
                    #Copy of data in CPU Memory
                    self.host_data = self.lib.asnumpy(pointset)
                #Copy of data in location memory
                self.data = self.lib.asarray(pointset, dtype=np.float32)


            if(self.location == "CPU"): 
                self.host_data = self.data

            self.dim = self.host_data.shape[1]

            if (self.local_size == 0):
                self.empty = True
            else:
                self.empty = False

        else:
            self.empty= True
            self.local_size = 0
            self.gids = self.lib.asarray([])

            if(self.sparse):

                if(self.location == "GPU"):
                    #Copy of CPU Memory
                    local_data = np.asarray([], dtype=np.float32)
                    local_indices = np.asarray([], dtype=np.int32)
                    local_indptr = np.asarray([], dtype=np.float32)
                    
                    self.host_data = sp.csr_matrix( (local_data, local_indicies, local_indptr) )

                #Copy of location memory
                local_data = self.lib.asarray([], dtype=np.float32)
                local_indices = self.lib.asarray([], dtype=np.int32)
                local_indptr = self.lib.asarray([], dtype=np.float32)
                
                self.data = self.lib.csr_matrix( (local_data, local_indicies, local_indptr) )
            else:
                if(self.location == "GPU"):
                    #Copy of data in CPU Memory
                    self.host_data = np.asarray([], dtype=np.float32)
                #Copy of data in location memory
                self.data = self.lib.asarray([], dtype=np.float32)

        #Assumes all trees have the same location and (dense/sparse)ness        
        Primitives.set_env(self.location, self.sparse)

        temp_local_size = np.array(self.local_size, dtype=np.int32)
        temp_global_size = np.array(0, dtype=np.int32)
        self.comm.Allreduce(temp_local_size, temp_global_size, op=MPI.SUM)
        self.size = temp_global_size

        if(self.location=="CPU"):
            self.host_data = self.data

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

        #Find out the maximum number of levels required
        #TODO: Fix definition of leafsize to make this proper. Can often overestimate level by one.
        self.levels = int(min( np.ceil(np.log2(np.ceil(self.size/self.leafsize))), self.levels ))
        self.dist_levels = int(np.floor(np.log2(self.comm.Get_size())))

        
        

        #Create nodelist to store nodes in binary tree array order
        #Nodelist is stored on the CPU
        N = 2 ** int(self.levels+1) - 1
        self.nodelist = [None] * N

        rank = self.comm.Get_rank()
        if rank == 0:
            vectors = np.random.rand(self.dim, self.levels)
            vectors = np.array(vectors, dtype=np.float32)
            qr_t = time.time()
            vectors = np.linalg.qr(vectors)[0]
            qr_t = time.time() - qr_t

            print("QR Time", qr_t)
            print(rank, vectors.shape)
        else:
            ld = min(self.dim, self.levels)
            vectors = np.zeros( (self.dim, ld), dtype=np.float32)

        print(rank, "Before BCAST", vectors.shape) 
        self.comm.Bcast(vectors, root=0)
        print(rank, "After BCAST", vectors.shape)
        print(rank, "After BCAST", vectors)

        vectors = vectors.T

        self.vectors = self.lib.array(vectors, dtype=np.float32)

        self.dist_vectors = vectors[:self.dist_levels, :]
        print(rank, "DIST_VECTORS", self.dist_vectors.shape)

        #TODO: Send stream of random integers for levels > dim case


        #self.size_list = [[N]]
        #self.offset_list = [[0]]
        #self.host_offset_list = [[0]]

        #Precompute the node sizes at each level
        #for level in range(0, self.levels):
        #    level_size_list = []
        #    for n in self.size_list[level]:
        #        level_size_list.append(np.floor(n/2))
        #        level_size_list.append(np.ceil(n/2))
        #    self.size_list.append(level_size_list)

        #for level in range(0, self.levels):
        #    self.size_list[level].insert(0, 0)

        #Precompute the offset for each node
        #for level in range(0, self.levels):
        #    self.offset_list.append(self.lib.cumsum(self.size_list[level]))
        #    self.host_offset_list.append(np.cumsum(self.size_list[level]))

        #print(self.offset_list)
        
        #Copy over from host data to data
        #TODO: Move this to after first log2(nGPU) partitions to support multiGPU case        
        """
        if (self.sparse):
           # #Copy of data in location Memory
            local_data = self.lib.asarray(self.host_data.data, dtype=np.float32)
            local_indices = self.lib.asarray(self.host_data.indices, dtype=np.int32)
            local_indptr = self.lib.asarray(self.host_data.indptr, dtype=np.int32)
            self.data = self.lib.csr_matrix( (local_data, local_indicies, local_indptr) )

            workspace_data = self.lib.zeros(self.size, dtype=np.float32)
            workspace_indices = self.lib.zeros(self.size, dtype=np.int32)
            workspace_indptr = self.lib.zeros(self.size+1, dtype=np.int32)
            workspace = (workspace_data, workspace_indices, workspace_indptr)
            

        else:
            temp = self.lib.asarray(self.host_data, dtype=np.float32)
            self.data = self.lib.zeros((2*self.size, self.dim), dtype=np.float32)
            self.data[:self.size, :] = temp
            
            workspace = self.lib.zeros(self.size, dtype=np.float32)

        #Begin construction

        #TODO: Parallelize this construction ? (Not a priority)
        size_list = [[N]]
        offset_list = [[0]]
        host_offset_list = [[0]]

        #Precompute the node sizes at each level
        for level in range(1, self.levels):
            level_size_list = []
            for n in size_list[level]:
                level_size_list.append(np.floor(n/2))
                level_size_list.append(np.ceil(n/2))
            size_list.append(level_size_list)

        #Precompute the offset for each node
        for level in range(1, self.levels):
            offset_list.append(self.lib.cumsum(size_list[level]))
            host_offset_list.append(np.cumsum(size_list[level]))


        #How many batch levels to do
        iters = self.levels/self.batch_levels        

        #Compute the projections, split the data, and form the nodes
        root = self.Node(self, idx=0, level=0, size=self.size, offset=0, gids=self.gids)
        nodelist[0] = root
        for batch in range(1, self.levels):
            self.projection = np.random.rand((self.batch_levels, self.dim), dtype=np.float32)
            self.data_offset = (level-1)%2  #which side of the self.data allocated, memory data is located on. Alternates with each iteration

            current_sizes = size_list[level]
            current_offsets = host_offset_list[level]

            data, medians, vectors = Primitives.build_level(self.data_offset, self.gids, self.data, offset_list[level], offset_list[level+1], self.dim, workspace)
            self.data = data

            start = 2**level - 1
            stop  = 2**(level+1) - 1
            level_size = stop - start
            for i in range(level_size):
                idx = start+i
                offset = current_offsets[i]
                size = current_sizes[i]
                node = self.Node(self, idx=idx, level=level, size=size, offset=offset, gids=self.gids[offset:offset+size])
                node.vector = vectors[i, ...]
                nodelist[idx] = node

        #Cleanup and set children
        for level in range(0, self.levels):
            start = 2**level - 1
            stop = 2**(level+1) -1
            for idx in range(start, stop):
                node = nodelist[idx]
                node.parent = nodelist[idx//2]
                node.set_children([nodelist[2*idx+1], nodelist[2*idx+2])
                if(level == self.levels-1):
                    node.isleaf=True
                  
        """
        
        #Distributed Level by Level construction
        #GOFMM Style (only support p = 2^x)
        comm = self.comm
        median_list = []

        for i in range(self.dist_levels):
            #Project and Reorder
            rank = comm.Get_rank()
            mpi_size = comm.Get_size()

            size = np.array(0, dtype=np.int32)
            local_size = np.array(self.local_size, dtype=np.int32)
            comm.Allreduce(local_size, size, op=MPI.SUM)

            a = i
            if i >= self.dim:
                a = random.randint(0, self.tree.dim-1)
            vector = self.dist_vectors[a, :]
            print(rank, vector.shape)

            proj_data = self.host_data @ vector

            lids = np.arange(self.local_size)
            lids = np.array(lids, dtype=np.int32)
            median, local_split = Primitives.dist_select(size/2, proj_data, lids, comm)
            self.gids = self.gids[lids] 
            self.host_data = self.host_data[lids, ...]

            print(rank, "original_proj", proj_data)
            print(rank, "max_o_p", np.max(proj_data))
            print(rank, "min_o_p", np.min(proj_data))
            print(rank, "local_median", np.median(proj_data))
            #Redistribute 
            #Pass local split to rank//2 if rank > comm.Get_size()
            
            if(rank >= mpi_size/2): #sending left, recv right
                send_offset = 0
                send_end = local_split
                recv_offset = 0
                send_target = rank - mpi_size//2
                send_tag = 11
                recv_tag = 22
                color = 1
            else: #recv left, send right
                send_offset = local_split
                send_end = self.local_size
                recv_offset = local_split
                send_target = rank + mpi_size//2
                send_tag = 22
                recv_tag = 11
                color = 0


            #list_sizes = np.zeros(mpi_size, dtype=np.int32)
            #list_sizes[rank] = send_size

            print(rank, "send_offset", send_offset, flush=True)
            print(rank, "send_end", send_end, flush=True)

            #Allocate memory for redistribute
            sending_ids = self.gids[send_offset:send_end]
            sending_data = self.host_data[send_offset:send_end]

            print(rank, "vector", vector, flush=True)

            send_req_ids = comm.isend(sending_ids, dest=send_target, tag=send_tag)
            recv_req_ids = comm.irecv(self.gids.size * self.gids.itemsize, source=send_target, tag=recv_tag)
            send_req_ids.wait()
            new_ids = recv_req_ids.wait()

            send_req_data = comm.isend(sending_data, dest=send_target, tag=send_tag)
            recv_req_data = comm.irecv(self.host_data.size * self.host_data.itemsize,source=send_target, tag=recv_tag)
            send_req_data.wait()
            new_data = recv_req_data.wait()


            print(rank, "old_data", self.data, flush=True)
            print(rank, "old_ids", self.gids, flush=True)

            print(rank, "new_ids", new_ids, flush=True)
            print(rank, "new_data", new_data, flush=True)


            print(rank, "median", median, flush=True)
            print(rank, "max_proj", np.max(new_data @ vector), flush=True)
            print(rank, "min_proj", np.min(new_data @ vector), flush=True)

            print(rank, "median", median, flush=True)
            print(rank, "max_proj_send", np.max(sending_data @ vector), flush=True)
            print(rank, "min_proj_send", np.min(sending_data @ vector), flush=True)

            recv_size = new_data.shape[0]
            self.host_data[recv_offset:(recv_offset)+recv_size] = new_data
            self.gids[recv_offset:(recv_offset)+recv_size] = new_ids
 
            median_list.append(median)           

            #Split communicator
            comm = comm.Split(color, rank)


        self.data = self.lib.array(self.host_data, dtype=np.float32)
 
        if self.sparse:
            print("Removed as it need changes soon")
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
                            root = self.Node(self, idx=0, level=self.dist_levels, size=self.local_size, gids=self.gids)
                            self.nodelist[0] = root

                        await T

                        #Build tree in n-order traversal
                        #TODO: Key area for PARLA Tasks
                        for level in range(self.levels):
                            start = 2**level -1
                            stop  = 2**(level+1) - 1
                            level_size = stop - start
                            data_size = self.size/2**level * 4
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
            self.ordered = False
        
        #Fix overestimate of tree levels (see #TODO above)
        if self.get_level(self.levels)[0] is None:
            self.levels -= 1
            self.nodelist = self.nodelist[:2**(self.levels+1)-1]


    def redistribute(neighbors):
        #Collect gids in order (TODO: Redo parla code so this is preserved)
        gidList = []
        for node in self.get_level(self.levels):
            gidList.append(node.gids)
        gids = np.concatenate(gidList)
        
        bins = defaultdict(list)
        for i in range(len(gids)):
            bins[gids[i]//self.local_size].append(i)
        print(bins)

        mpi_size = comm.Get_size()
        rank = comm.Get_rank()
 
        for i in range(mpi_size):
            if i != rank:
                sending_data = self.data[bins[i]]
                send_req_data = comm.isend(sending_data, dest=send_target, tag=send_tag)
                recv_req_data = comm.irecv(self.host_data.size * self.host_data.itemsize,source=send_target, tag=recv_tag)

        for i in range(mpi_size):
            send_req_data.wait()
            new_data = recv_req_data.wait()
             


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
            if self.tree.ordered:
                return self.tree.data[offset:offset+size, ...]
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
            #print("size", self.size)
            #print("off", self.offset)
            a = self.level
            if self.level >= self.tree.dim:
                a = random.randint(0, self.tree.dim-1)
            self.vector = self.tree.vectors[a, :]
            #self.vector = self.lib.random.rand(self.tree.dim, 1).astype(np.float32)
            #self.vector = self.vector/self.lib.linalg.norm(self.vector, 2)
            #print("DAT", self.tree.data[self.gids, ...].shape, flush=True)
            #print("vec", self.vector.shape, flush=True)
            self.local_ = self.tree.data[self.gids, ...] @ self.vector
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

            self.tree.nodelist[self.id] = self

            #Stop the split if the leafsize is too small, or the maximum level has been reached

            if (middle < self.tree.leafsize) or (self.level+1) > self.tree.levels:
                self.plane = None
                self.anchors = None
                self.vector = None
                self.isleaf=True
                return [None, None]

            if self.tree.location == "CPU":

                #project onto line (projection stored in self.local_)
                self.select_hyperplane()
                self.lids = self.lib.argpartition(self.local_, middle)  #parition the local ids

                self.gids = self.gids[self.lids]                  #partition the global ids

                #TODO: When replacing this with Hongru's kselect the above should be the same step in a key-value pair

                self.plane = self.local_[self.lids[middle]]       #save the splitting line

                self.cleanup()                                    #delete the local projection (it isn't required any more)

                #Initialize left and right nodes
                
                left = self.tree.Node(self.tree, level = self.level+1, idx = 2*self.id+1, size=middle, gids=self.gids[:middle])
                right = self.tree.Node(self.tree, level = self.level+1, idx = 2*self.id+2, size=int(self.size - middle), gids=self.gids[middle:])

                left.set_parent(self)
                right.set_parent(self)

                children = [left, right]
                self.set_children(children)

                return children

            elif self.tree.location == "GPU":

                #project onto line (projection is stored in self.local_)

                with stream:
                    a = self.level
                    if self.level >= self.tree.dim:
                        a = random.randint(0, self.tree.dim-1)
                    self.vector = self.tree.vectors[a, :]
                    self.local_ = self.tree.data[self.gids, ...] @ self.vector
                    self.lids = self.lib.argpartition(self.local_, middle)
                    self.gids = self.gids[self.lids]
                    self.plane = self.local_[self.lids[middle]]

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
            return Primitives.single_knn(self.gids, R, Q, k)

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

        N = self.size
        max_level = self.levels

        #Allocate space to store results
        neighbor_list = self.lib.zeros([N, k], dtype=np.int32)
        neighbor_dist = self.lib.zeros([N, k], dtype=np.float32)
        
        #get all leaf nodes
        leaf_nodes = self.get_level(max_level)
        n_leaves = len(leaf_nodes)

        #compute batchsize
        MAXBATCH = 2048
        n_leaves = len(leaf_nodes)
        batchsize = n_leaves if n_leaves < MAXBATCH else MAXBATCH 
        
        #Loop over all batches 
        iters = int(np.ceil(n_leaves/batchsize))

        #TODO: Make bookkeeping parallel
        print("STARTING BATCH:", iters)
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
                RList.append(leaf.get_reference())
                
            #print("Populated RList")
            #call batch routine

            #print(gidsList)
            #print("RList", RList[0].shape)
            #print(k)
            setup_t = time.time() - setup_t
            print("Setup time took ", setup_t)

            comp_t = time.time()
            NLL, NDL = Primitives.multileaf_knn(gidsList, RList, RList, k)
            comp_t = time.time() - comp_t
            print("Computation took ", comp_t)

            #print("Finished kernel call")

            copy_t = time.time()
            #populate results from temporary local objects
            j = 0;
            for leaf in leaf_nodes[start:stop]:
                idx = leaf.get_gids()
                lk = min(k, leaf.size-1)
                NL =  NLL[j]
                ND =  NDL[j]
                neighbor_list[idx, :] = NL[:, :]
                neighbor_dist[idx, :] = ND[:, :]
                j += 1
            copy_t = time.time() - copy_t
            print("Copy took ", copy_t)

            #print("Finished copy")
            batch_t = time.time() - batch_t
            print("BATCH TOOK:", batch_t)

        return neighbor_list, neighbor_dist


