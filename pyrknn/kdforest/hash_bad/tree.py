import random

from parla import Parla
from parla.array import copy, storage_size
from parla.cpu import cpu
from parla.tasks import *

import os
import time
import gc 

import numpy as np

from . import util
"""
#Shared State Q
Qlist = []
for k in range(10):
    v = np.random.rand(784, 784)
    v = np.asarray(v, dtype=np.float32, order='F')
    Q = np.linalg.qr(v)[0]
    Qlist.append(Q)

q_idx = 0
ql_idx = 0
"""

class RKDT:

    def __init__(self, forest, data, leafsize=512):
        self.id = id(self)
        self.forest = forest 
        self.built = False
        self.leafsize = leafsize
        self.data = data
        self.N = data.shape[0]
        self.d = data.shape[1]
        self.gids = np.arange(self.N, dtype=np.int32)

    def generate_vectors(self):
        timer = util.Profiler()
        vectors = np.random.randn(self.d, self.levels)
        vectors = np.asarray(vectors, dtype=np.float32, order="F")
        timer.push("Orthogonalize")
        vectors = np.linalg.qr(vectors)[0]
        timer.pop("Orthogonalize")

        if self.levels > self.d:
            spillover = np.random.randint(low=0, high=self.d, size=self.levels-self.d, dtype=np.int32)
        
        self.vectors = vectors.T

    def query(self, q):
        idx = 0
        for l in range(self.levels):
            current = self.nodelist[idx]
            idx = current.query(q)
        return idx

    def batch_query(self, Q):
        N = Q.shape[0]
        d = self.d

        idx = np.zeros(N)

        for i in range(N):
            idx = query(Q[i,:])
        return idx

    def aknn_all(self, k):
        """Perform approximate all knn search

        Arguments:
            k -- The number of nearest neighbors (Require k < leafnodesize, otherwise it is padded with inf values)

        Return:
            (neighbor_list , neigbor_dist)
            neighbor_list -- |Q| x k list of neighbor gids. In local ordering og the query points.
            neighbor_dist -- the corresponding distances
        """
        timer = util.Profiler()

        timer.push("AKNN")

        rank = 0#self.comm.Get_rank()

        N = self.N
        max_level = self.levels

        #Allocate space to store results
        neighbor_list = np.zeros([N, k], dtype=np.int32)
        neighbor_dist = np.zeros([N, k], dtype=np.float32)

        #get all leaf nodes
        leaf_nodes = self.get_level(self.levels)
        n_leaves = len(leaf_nodes)

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
            NLL, NDL, out = util.multileaf_knn(gidsList, RList, RList, k)
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
                neighbor_list[idx, :] = NL[:, :]
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
        
    def search(self, k):

        N = self.N

        #Allocate space to store results
        neighbor_list = np.zeros([N, k], dtype=np.int32)
        neighbor_dist = np.zeros([N, k], dtype=np.float32)

        #get all leaf nodes
        leaf_nodes = self.get_level(self.levels)
        n_leaves = len(leaf_nodes)

        for leaf in leaf_nodes:
            gl = leaf.gids
            rl = np.copy(leaf.get_reference())
            NL, ND = util.single_knn(gl, rl, rl, k)
            idx = leaf.get_gids()
            lk = min(k, leaf.size-1)
            neighbor_list[idx, :] = NL[:, :]
            neighbor_dist[idx, :] = ND[:, :]
        
        return neighbor_list, neighbor_dist

    def direct(self, Q, k):
        root = self.nodelist[0]
        return root.direct(Q, k)

    def get_level(self, level):
       start = 2**level - 1
       stop = 2**(level+1) - 1
       return self.nodelist[start:stop]

    def __str__(self, verbose=False):
        if self.built:
            msg = 'Built tree of depth '+str(self.levels)
        else:
            msg = 'Unbuilt tree'

    def build(self):


        self.levels = (int)(np.ceil(np.log2(self.N/self.leafsize)))

        self.generate_vectors()

        self.size_list = [[self.N]]
        self.offset_list = [[0]]

        #Precompute node size
        for level in range(0, self.levels+1):
            level_size_list = []
            for n in self.size_list[level]:
                level_size_list.append(np.floor(n/2))
                level_size_list.append(np.ceil(n/2))
            self.size_list.append(level_size_list)

        for level in range(0, self.levels+1):
            self.size_list[level].insert(0, 0)
        
        #Precompute internal offset

        for level in range(0, self.levels+1):
            self.offset_list.append(np.cumsum(self.size_list[level]))
        
        self.nodelist = [None] * (2**(self.levels+2)-1)

        root = Node(self, idx=0, level=0, size=self.N, gids=self.gids)
        self.nodelist[0] = root
        

        for level in range(0, self.levels):
            start = 2**level - 1
            stop = 2**(level+1) - 1
            level_size = stop - start
            for i in range(level_size): 
                current = self.nodelist[start+i]
                if current is not None:
                    children = current.split()
                    children = list(filter(None, children))
                    for child in children:
                        idx = child.get_idx()
                        self.nodelist[idx] = child

        while self.get_level(self.levels)[0] is None:
            self.levels -= 1
            self.nodelist = self.nodelist[:2**(self.levels+1)-1]
        self.built = True 

    def hash(index, label="id"):
        leaf_nodes = self.get_level(self.levels)
        i = 1
        for leaf in leaf_nodes:
            if label == "id":
                self.forest.hash[leaf.gids, index] = np.ones(len(leaf.gids)) * i
            elif label == "mean":
                self.forest.hash[leaf.gids, index] = np.ones(len(leaf.gids)) * leaf.center()
            i += 1   

class Node:

    def __init__(self, tree, idx=0, level=0, size=0, gids=None):
        self.tree = tree
        self.idx = idx
        self.level = level
        self.size = size
        self.gids = gids
        self.vector = None
        self.anchors = None
        self.isleaf = True 
        self.parent = None
        self.children=[None, None]
        self.verbose=True
        self.offset = int(self.tree.offset_list[self.level+1][idx-2**self.level+1])
        self.plane = None

    def __str__(self):
        """Overloading the print function for a RKDT.Node class"""
        if self.verbose:
            msg = 'Node: ' + str(self.idx) + ' at level '+ str(self.level) + ' of size ' + str(self.size)
            msg += '\nLeaf:' + str(self.isleaf)
            msg += '\nBelonging to tree with id ' + str(self.tree.id)
            msg += '\nOffset: '+str(self.offset)
            msg += '\nSplitting Line: '+str( (self.plane, self.vector) )
            msg += '\nContains gids:' +str(self.gids)
            #msg += '\nData:'+str(self.tree.data[self.gids, ...])
            msg += '\n--------------------------'
        else:
            msg = 'Node: ' + str(self.idx) + ' at level '+ str(self.level) + ' of size ' + str(self.size)
        return msg

    def get_idx(self):
        return self.idx
    
    def get_gids(self):
        return self.gids

    def get_reference(self):
        return self.tree.data[self.offset:self.offset+self.size, ...]

    def set_children(self, nodelist):
        self.children = nodelist
        self.isleaf if all([child is None for child in nodelist]) else False
    
    def set_parent(self, node):
        self.parent = node

    def center(self):
        data = self.get_reference()
        center = np.mean(data, axes=1)
        return center 

    def hyperplane(self):
        a = self.level
        if a >= self.tree.d:
            a = np.random.randint(0, self.tree.d-1)
        self.vector = self.tree.vectors[a, :]
        #self.vector = np.asarray(np.random.randn(self.tree.d, 1), dtype=np.float32)
        self.vector = self.vector/ np.linalg.norm(self.vector)
        self.local_ = self.tree.data[self.offset:self.offset+self.size, ...] @ self.vector
 
    def clean(self):
        del self.local_ 

    def split(self):

        middle = int(self.size//2)
        self.tree.nodelist[self.idx] = self

        if (middle < self.tree.leafsize):
            self.plane = None
            self.anchors = None
            self.vector = None
            self.isleaf = True
            return [None, None]

        self.hyperplane()
        self.lids = np.argpartition(self.local_, middle)
        self.tree.gids[self.offset:self.offset+self.size] = self.gids[self.lids]
        self.gids = self.tree.gids[self.offset:self.offset+self.size]
        self.plane = self.local_+[self.lids[middle]] # save the splitting median
        self.clean()

        left = Node(self.tree, level=self.level+1, idx=2*self.idx+1, size=middle, gids=self.gids[:middle])
        right = Node(self.tree, level=self.level+1, idx=2*self.idx+2, size=int(self.size - middle), gids=self.gids[middle:])
        
        left.set_parent(self)
        right.set_parent(self)
        children = [left, right]
        self.set_children(children)

        local_data = self.tree.data[self.offset:self.offset+self.size]
        self.tree.data[self.offset:self.offset+self.size] = local_data[self.lids]

        del local_data
        del self.lids

        return children

    def direct(self, Q, k):
        R = self.get_reference()
        lids = np.arange(self.size, dtype=np.int32)
        results = util.single_knn(self.gids, R, Q, k)
        results = util.merge_neighbors(results, results, k)
        return results




    
