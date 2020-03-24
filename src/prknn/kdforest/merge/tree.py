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

from collections import defaultdict

class RKDT:
    """Class for Randomized KD Tree Nearest Neighbor Searches"""

    verbose = False #Note: This is a static variable shared by all instances

    def __init__(self, levels=0, leafsize=None, pointset=None, location="CPU"):
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

        if(self.location == "CPU"):
            self.lib = np
        elif(self.location == "GPU"):
            self.lib = cp 

        if (pointset is not None):
            self.size = len(pointset)           #the number of points in the pointset
            self.gids = self.lib.arange(self.size, dtype=np.int32)    #the global ids of the points in the pointset (assign original ordering)
            self.data = self.lib.asarray(pointset, dtype=np.float32)

            if (leafsize is None):              #if no leafsize is given, assume this is a degenerate tree (only root)
                self.leafsize = self.size
            if (self.size == 0):
                self.empty = True
            else:
                self.empty = False
        else:
            self.empty= True
            self.size = 0
            self.data = self.lib.asarray([])
            self.gids = self.lib.asarray([])
        
        Primitives.set_env(self.location)
        self.built=False

    def set(pointset=None, leafsize=None, levels=None):
        """Set or redefine the key values of a RKDT

            Keyword arguments:
                levels -- maximum number of levels in the tree
                leafsize -- Leaves of size (2*leafsize + 1) will not be split
                pointset -- the N x d dataset of N d dimensional points in Rn
        """
        if self.built:
            raise ErrorType.InitializationError('You cannot call set on a tree that has already been built')

        if (pointset is not None):
            self.data = self.lib.asarray(pointset)
            self.size = len(pointset)
            if (self.size > 0):
                self.empty= False
            else:
                self.empty=True
        if (leafsize is not None):
            assert(leafsize >= 0)
            self.leafsize = leafsize
        if (levels is not None):
            assert(levels >= 0)
            self.levels = levels

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

       #Create nodelist to store nodes in binary tree array order
        N = 2 ** int(self.levels+1) - 1
        self.nodelist = [None] * N

        if self.location == "CPU":
           #Create nodelist to store nodes in binary tree array order

            with Parla():
                @spawn(placement = cpu)
                async def build_tree():
                    T = TaskSpace()
                    #Create the root node
                    @spawn(T[0], placement = cpu)
                    def create_root():
                        root = self.Node(self, idx=0, level=0, size=self.size, gids=self.gids)
                        self.nodelist[0] = root
                    
                    #Build tree in in-order traversal
                    #TODO: Key area for PARLA Tasks
                    for level in range(self.levels):
                        start = 2**level -1
                        stop  = 2**(level+1) - 1
                        level_size = stop - start
                        data_size = self.size/2**level * 4
                        for i in range(level_size):
                            @spawn(T[start+i+1], [T[0], T[int((start+i+1)/2)]], placement = cpu, memory=data_size)
                            def create_children_task():
                                current_node = self.nodelist[start+i]
                                if current_node is not None:
                                    children = current_node.split()
                                    children = list(filter(None, children))
                                    for child in children:
                                        idx = child.get_id()
                                        self.nodelist[idx] = child
                    await T
                    self.built=True
        elif self.location == "GPU":
            root = self.Node(self, idx=0, level=0, size=self.size, gids=self.gids)
            self.root = root
            root.split(cp.cuda.Stream(non_blocking=True))
            self.built=True

        #Fix overestimate of tree levels (see #TODO above)
        if self.get_level(self.levels)[0] is None:
            self.levels -= 1

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
            self.isleaf = True
            self.parent = None
            self.children = [None, None]
            self.anchors = None
            self.plane = None
            self.vector = None
            
            self.lib = self.tree.lib

        def __str__(self):
            """Overloading the print function for a RKDT.Node class"""

            if self.verbose:
                msg = 'Node: ' + str(self.id) + ' at level '+ str(self.level) + ' of size ' + str(self.size)
                msg += '\nLeaf:' + str(self.isleaf)
                msg += '\nBelonging to tree with id ' + str(self.tree.id)
                msg += '\nAnchor points are: '
                anchors = self.anchors if self.anchors is not None else []
                for anchor in anchors:
                    msg += '\ngid:'+str(anchor)+' value: '+str(self.tree.data[anchor, ...])
                msg += '\nSplitting Line: '+str(self.plane)
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

        def data(self):
            """Return the size x d array of the owned pointset. Note this produces a copy."""
            return self.tree.data[self.gids, ...]

        def get_reference(self):
            """Return the size x d array of the owned pointset. Note this produces a copy."""
            return self.tree.data[self.gids, ...]

        def cleanup(self):
            """Delete the local projection onto the random line"""
            self.local_ = []

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

        def select_hyperplane(self):
            """Select the random line and project onto it.

                Algorithm:
                    Select 2 random points owned by this node.
                    Compute the distance from all other points to these two 'anchor points'
                    The projection local_ is their difference.

                    This computes <x, (a_1 - a_2)>
            """
            #TODO: Replace with gpu kernel
            self.anchors = self.lib.random.choice(self.gids, 2, replace=False)
            dist = Primitives.distance(self.tree.data[self.gids, ...], self.tree.data[self.anchors, ...])
            self.local_ = dist[0] - dist[1]

        def average(self, idx=0):
            """Return the average of points in the node along a specified axis (0 < idx < d-1)

            Keyword Arguments:
                idx -- axis to compute average along. If idx < 0 compute with the saved local projection onto anchor points
            """

            if (idx >= 0):
                return self.lib.mean(self.tree.data[self.gids, idx])
            else:
                return self.lib.mean(self.local_)

        def median(self, idx=0):
            """Return the median of points in the node along a specified axis (0 < idx < d-1)

            Keyword Arguments:
                idx -- axis to compute median along. If idx < 0 compute with the saved local projection onto anchor points
            """

            if (idx >= 0):
                return self.lib.median(self.tree.data[self.gids, idx])
            else:
                return self.lib.median(self.local_)

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
                    self.vector = self.lib.random.random((self.tree.data.shape[1]), dtype='float32')
                    self.local_ = self.lib.dot(self.tree.data[self.gids, ...], self.vector)
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
            R = self.tree.data[self.gids, ...]
            return Primitives.single_knn(self.gids, R, Q, k)

        def knn_all(self, k):
            """
            Perform an exact exhaustive all-knn search in the node. O(size x gids x d)

            Arguments:
                k -- number of nearest neighbors (Limitation: k < leafsize)
            """
            R = self.tree.data[self.gids, ...]
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

            #compute distance to anchors
            q = q.reshape((1, len(q)))
            if self.tree.location == "CPU":
                dist = Primitives.distance(q, self.tree.data[self.anchors, ...])
                dist = dist[0] - dist[1]
            elif self.tree.location == "GPU":
                dist = self.lib.dot(q, self.vector) 

            #compare against splitting plane
            return 2*self.id+1 if dist < self.plane else 2*self.id+2

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

        N, d = Q.shape
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
        neighbor_list = self.lib.full([N, k], self.lib.inf)
        neighbor_dist = self.lib.full([N, k], self.lib.inf)
        
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
        neighbor_list = self.lib.full([N, k], np.inf)
        neighbor_dist = self.lib.full([N, k], np.inf)
        
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
                #print(NLL)
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


