#import numpy as np
from collections import defaultdict
import threading
import numpy as np
import cupy as cp
import util

def split_node(node):
    if node is None:
        return
    node.split()
    return 

class RKDT:
    """Class for Randomized KD Tree Nearest Neighbor Searches"""
    """The split method in this file is done with synchronization at each level"""

    verbose = False #Note: This is a static variable shared by all instances

    def __init__(self, libpy, levels=0, leafsize=1024, pointset=None):
        """Initialize  Randomized KD Tree

            Keyword arguments:
                levels -- maximum number of levels in the tree
                leafsize -- Leaves of size (2*leafsize + 1) will not be split
                pointset -- the N x d dataset of N d dimensional points in Rn
        """
        self.libpy = libpy
        self.id = id(self)                      #unique id for instance of this class
        self.levels = levels
        self.leafsize = leafsize
        self.nodelist = []
        #assert(pointset is not None)
        self.data = [pointset, cp.zeros(pointset.shape)]
        self.size = len(pointset)
        self.proj_array = cp.zeros(self.size)
        self.index_array = cp.zeros(self.size,dtype='int32')
        self.entry_shape = pointset[0].shape
        self.built=False

    '''
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
            self.data = self.libpy.asarray(pointset)
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
    '''

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
            leafsize -- Leaves of size = 2048 will not be split
        """

        #Various error checking methods to make sure tree is initialized properly
        if self.built:
            raise ErrorType.InitializationError('You cannot call build on a tree that has already been built')
        '''
        if self.empty:
            raise ErrorType.InitializationError('Cannot build an empty tree')
        if self.size < 0:
            raise ErrorType.InitializationError('Invalid size parameter: Cannot build a tree of size '+str(self.size))
        if self.leafsize <= 0:
            raise ErrorType.InitializationError('Invalid leaf size parameter: Cannot build a tree with leaf size '+str(self.leafsize))
        if self.levels < 0:
            raise ErrorType.InitializationError('Invalid max levels parameter: Cannot build a tree of '+str(self.levels)+' levels')
        '''
        #Create the root node
        root = self.Node(self.libpy, self.entry_shape, self, idx=0, level=0, size=self.size, start=0, end=self.size)

        data_size = self.size
        start = 0
        stride = 1
        streams = [] # We only need 16 streams max
        for i in range(16):
            streams.append(cp.cuda.Stream(null=False, non_blocking=True))
        while data_size > 1024:
            '''
                For the first group of 16 nodes, we should get the stream and call node split.
                For the later groups of 16 nodes, we first call stream synchronize then call 
                node split. 

                Node split function should add itself to self.nodelist.  
            '''
            if stride <=8:
                for i in range(stride):
                    node = self.nodelist[start+i]
                    node.split(streams[i])
                for i in range(stride):
                    streams[i].synchronize()
            else:
                nelem = stride//16
                for i in range(16):
                    for j in range(nelem):
                        node = self.nodelist[start + j + i*nelem]
                        node.split(streams[i])

            data_size //= 2
            start += stride
            stride *= 2

        for stream in streams:
            stream.synchronize()
        self.built=True

    class Node:

        verbose = False

        def __init__(self, libpy, entry_shape, tree, idx=0, level=0, size=0, start=None, end=None):
            """Initalize a member of the RKDT.Node class

            Arguments:
                tree -- the owning RKDT (used for grabbing data from the pointset)

            Keyword Arguments:
                idx -- the binary tree array order index
                level -- the level in the tree of the node (root = level 0)
                size -- the number of points that this node corresponds to
                gids -- the list of global indicies for the owned points
            """
            
            self.entry_shape = entry_shape
            self.libpy = libpy
            self.tree = tree
            self.id = idx
            self.level = level
            self.size = size
            #self.gids = gids
            self.isleaf = True
            self.parent = None
            self.children = [None, None]
            self.plane = [None, 0.0]
            self.tree.nodelist.append(self)
            assert(start is not None)
            self.start = start
            self.end = end

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

        def cleanup(self):
            """Delete the local projection onto the random line"""
            self.local_ = []

        def set_right_child(self, node):
            """Set the 'pointer' to the right child. (Points should be > split)"""
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

        '''
        def select_hyperplane(self):
            """Select the random line and project onto it.

                Algorithm:
                    Select 2 random points owned by this node.
                    Compute the distance from all other points to these two 'anchor points'
                    The projection local_ is their difference.

                    This computes <x, (a_1 - a_2)>
            """
            #TODO: Replace with gpu kernel
            self.anchors = self.libpy.random.choice(self.gids, 2, replace=False)
            dist = util.distance(self.tree.data[self.gids, ...], self.tree.data[self.anchors, ...])
            self.local_ = dist[0] - dist[1]
        '''

        def average(self, idx=0):
            """Return the average of points in the node along a specified axis (0 < idx < d-1)

            Keyword Arguments:
                idx -- axis to compute average along. If idx < 0 compute with the saved local projection onto anchor points
            """

            if (idx >= 0):
                return self.libpy.mean(self.tree.data[self.gids, idx])
            else:
                return self.libpy.mean(self.local_)

        def median(self, idx=0):
            """Return the median of points in the node along a specified axis (0 < idx < d-1)

            Keyword Arguments:
                idx -- axis to compute median along. If idx < 0 compute with the saved local projection onto anchor points
            """

            if (idx >= 0):
                return self.libpy.median(self.tree.data[self.gids, idx])
            else:
                return self.libpy.median(self.local_)

        def split(self, stream):
            """Split a node and assign both children.

            Return value:
                children -- [left, right] containing the left and right child nodes respectively

            Algorithm:
                Project onto line. (un-normalized in the current implementation)
                Split at median of projection.
                Partition gids and assign to children. left < median, right > median
            """

            middle = self.size//2

            #Stop the split if the leafsize is too small
            if (middle < self.tree.leafsize):
                '''if (middle < self.tree.leafsize) or (self.level+1) > self.tree.levels:'''
                self.plane = None
                self.isleaf=True
                return [None, None]

            '''
            if (self.level == 0 or self.level == 1 or self.level == 2):
                mem_pool = cp.get_default_memory_pool()
                print('before creating stream, used bytes at level', self.level, 'are ', mem_pool.used_bytes())
                print('before creating stream, total bytes at level ', self.level,' are ', mem_pool.total_bytes())
            '''
        
            #project onto line
            data_index1 = self.level % 2
            data_index2 = (data_index1 + 1) % 2
            with stream:
                cp.random.RandomState(1001+self.id)
                self.plane[0] = self.libpy.random.random((self.entry_shape),dtype='float32')
                self.tree.proj_array[self.start:self.end] = self.libpy.dot(self.tree.data[data_index1][self.start:self.end], self.plane[0])
                self.tree.index_array[self.start:self.end] = self.libpy.argpartition(self.tree.proj_array[self.start:self.end], middle)
                self.plane[1] = self.tree.proj_array[self.start + self.tree.index_array[self.start+middle]]
                #Copy data here
                self.tree.data[data_index2][self.start:self.start+middle] = self.tree.data[data_index1][self.tree.index_array[self.start:self.start+middle]]
                self.tree.data[data_index2][self.start+middle:self.end] = self.tree.data[data_index1][self.tree.index_array[self.start+middle:self.end]]

            #Initialize left and right nodes
            left = self.tree.Node(self.libpy, self.entry_shape, self.tree, level = self.level+1, idx = 2*self.id+1, size=middle, start=self.start, end=self.start+middle)
            right = self.tree.Node(self.libpy, self.entry_shape, self.tree, level = self.level+1, idx = 2*self.id+2, size=int(self.size - middle),start=self.start+middle,end=self.end)
            
            left.set_parent(self)
            right.set_parent(self)

            children = [left, right]
            self.set_children(children)

            '''
            if (self.level == 0 or self.level == 1 or self.level == 2):
                mem_pool = cp.get_default_memory_pool()
                print('after creating stream and deleting, used bytes at level', self.level, 'are ', mem_pool.used_bytes())
                print('after creating stream and deleting, total bytes at level ', self.level,' are ', mem_pool.total_bytes())
            '''
           
            return children

        def knn(self, Q, k):
            """
            Perform an exact exhaustive knn query search in the node. O(size x gids x d)

            Arguments:
                Q -- N x d matrix of query points
                k -- number of nearest neighbors
            """
            R = self.tree.data[self.gids, ...]
            return util.direct_knn(self.gids, R, Q, k)

        def exact_all_nearest_neighbors(self, k):
            """
            Perform an exact exhaustive all-knn search in the node. O(size x gids x d)

            Arguments:
                k -- number of nearest neighbors (Limitation: k < leafsize)
            """
            R = self.tree.data[self.gids, ...]
            return util.direct_knn(self.gids, R, R, k)

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
            q = q.reshape((1, q.shape[0]))
            dist = util.distance(q, self.tree.data[self.anchors, ...])
            dist = dist[0] - dist[1]
            print(q.shape)
            print(dist.shape)
            print("1x1")
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
        idx = self.libpy.zeros(N)

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
        bins = self.query_bin(Q)
        neighbor_list = self.libpy.full([N, k], self.libpy.inf)
        neighbor_dist = self.libpy.full([N, k], self.libpy.inf)

        #TODO: Key area for PARLA Tasks

        for leaf in bins:
            idx = bins[leaf]
            node = self.nodelist[leaf]
            lk = min(k, node.size-1) #TODO: Ascend tree if k > leafsize?
            lneighbor_list, lneighbor_dist = node.knn(Q[idx, ...], lk)
            neighbor_list[idx, :lk] = lneighbor_list
            neighbor_dist[idx, :lk] = lneighbor_dist

        return neighbor_list, neighbor_dist

    def all_nearest_neighbor(self, k):
        """Perform approximate all knn search at the root level of the tree

        Arguments:
            k -- The number of nearest neighbors (Require k < leafnodesize, otherwise it is padded with inf values)

        Return:
            (neighbor_list , neigbor_dist)
            neighbor_list -- |Q| x k list of neighbor gids. In local ordering og the query points.
            neighbor_dist -- the corresponding distances
        """

        N = self.size
        neighbor_list = self.libpy.full([N, k], -1)
        neighbor_dist = self.libpy.full([N, k], -1.0)
        max_level = self.levels

        #TODO: Key area for PARLA tasks
        for leaf in self.get_level(max_level):
            idx = leaf.gids
            lk = min(k, leaf.size-1) #TODO: Ascend tree if k > leafsize?
            lneighbor_list, lneighbor_dist = leaf.knn(self.data[idx, ...], lk)
            neighbor_list[idx, :lk] = lneighbor_list
            neighbor_dist[idx, :lk] = lneighbor_dist
        return neighbor_list, neighbor_dist


