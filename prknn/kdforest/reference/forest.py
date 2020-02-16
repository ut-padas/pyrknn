from . import error as ErrorType
from . import util as Primitives
from .tree import *

import numpy as np
from collections import defaultdict

class RKDForest:
    """Class for a collection of RKD Trees for Nearest Neighbor searches"""

    verbose = False #Note: This is a static variable

    def __init__(self, ntrees=1, levels=0, leafsize=None, pointset=None):
        """Initialize Randomized KD Forest.

        Keyword Arguments:
            ntrees -- number of trees to generate in the forest
            levels -- the maximum number of levels for all trees
            leafsize -- nodes are not split below size (2*leafsize +1)
            pointset -- the pointset for the nearest neighbor searches
        """
        if levels < 0:
            raise ErrorType.InitializationError('Invalid max levels parameter: Cannot build trees with '+str(levels)+' levels')

        if ntrees < 0:
            raise ErrorType.InitializationError('Invalid ntrees parameter: '+str(ntrees)+' ntrees')

        self.id = id(self)
        self.levels = levels
        self.leafsize = leafsize
        self.ntrees = ntrees

        if (pointset is not None):
            self.size = len(pointset)
            self.gids = np.arange(self.size)
            self.data = np.asarray(pointset)
            if (leafsize is None):
                self.leafsize = self.size
            if (self.ntrees == 0):
                self.empty = True
            else:
                self.empty = False

            if self.leafsize < 0:
                raise ErrorType.InitializationError('Invalid leafsize parameter: '+str(self.leafsize))
        else:
            self.empty= True
            self.ntrees = 0
            self.data = np.asarray([])
            self.gids = np.asarray([])

        self.built=False


    def build(self):
        """Build the RKD Forest"""

        #checking parameters to ensure valid initialization
        if self.built:
            raise ErrorType.InitializationError('You cannot build a forest that has already been built. Please clear() or use grow()')
        if self.empty:
            raise ErrorType.InitializationError('Cannot build an empty forest')
        if self.ntrees < 0:
            raise ErrorType.InitializationError('Invalid size parameter: Cannot build a forest of size '+str(self.size))
        if self.leafsize <= 0:
            raise ErrorType.InitializationError('Invalid leaf size parameter: Cannot build trees with leaf size '+str(self.leafsize))
        if self.levels < 0:
            raise ErrorType.InitializationError('Invalid max levels parameter: Cannot build trees with '+str(self.levels)+' levels')

        #TODO: Key area for PARLA Tasks

        self.forestlist = [None]*self.ntrees

        for i in range(self.ntrees):
            tree = RKDT(pointset=self.data, levels=self.levels, leafsize=self.leafsize)
            tree.build()
            self.forestlist[i] = tree

        self.built=True

    def knn(self, Q, k):
        """Perform an exact exhaustive knn search using the root node of the first tree.

        Arguments:
            Q -- N x d query matrix
            k -- number of nearest neighbors (req. k < leafsize)

        Return:
            (neighbor_list , neigbor_dist)
            neighbor_list -- |Q| x k list of neighbor gids. In local ordering og the query points.
            neighbor_dist -- the corresponding distances
        """

        first = self.forestlist[0]
        return first.knn(Q, k)

    def aknn(self, Q, k, verbose=False):
        """Perform an approximate knn search over all trees, merging results.

        Arguments:
            Q -- N x d query matrix
            k -- number of nearest neighbors (req. k < leafsize)

        Return:
            (neighbor_list , neigbor_dist)
            neighbor_list -- |Q| x k list of neighbor gids. In local ordering og the query points.
            neighbor_dist -- the corresponding distances
        """
        v = (self.verbose or verbose)
        result = None

        #TODO: Key Area for PARLA Tasks

        for tree in self.forestlist:
            neighbors = tree.aknn(Q, k)

            if result is None:
                result = neighbors
            else:
                result, changes = Primitives.merge_neighbors(result, neighbors, k)

            #TODO: Delay merge until after all searches (increase storage to O(|Q| x k ) x ntrees but increase potential task parallelism ?

        return result

    def all_nearest_neighbor(k):
        """Perform an approximate all knn search over all trees in forest, merging results.

        Arguments:
            Q -- N x d query matrix
            k -- number of nearest neighbors (req. k < leafsize)

        Return:
            (neighbor_list , neigbor_dist)
            neighbor_list -- |Q| x k list of neighbor gids. In local ordering og the query points.
            neighbor_dist -- the corresponding distances
        """
        v = (self.verbose or verbose)
        result = None

        #TODO: Key Area for PARLA Tasks

        for tree in self.forestlist:
            neighbors = tree.all_nearest_neighbor(k)

            if result is None:
                result = neighbors
            else:
                result, changes = Primitives.merge_neighbors(result, neighbors, k)

            #TODO: Delay merge until after all searches (increase storage to O(|Q| x k ) x ntrees but increase potential task parallelism ?

        return result

    def grow(l):
        """Add l more trees to the forest"""

        #TODO: Key Area for PARLA Tasks
        for i in range(l):
            tree = RKDT(pointset=self.data, levels=self.levels, leafsize=self.leafsize)
            self.treelist.append(tree)
        self.ntrees = len(self.treelist)

    def clear():
        """Clear the forest."""

        if self.built:
            self.forestlist = []
        self.data = np.asarray([])
        self.ntrees = 0
        self.size = 0
        self.empty = True

    def check_accuracy(self, Q=None, k=None):
        """Perform a comparison of knn accuracy between the approximate and exact search.

        Keywork Arguments:
            Q -- N x d query matrix
            k -- number of nearest neighbors (req. k < leafsize)

        if k not set, uses leafsize - 1
        if Q not set, performs all nearest neighbor search

        """
        if k is None:
            k = self.leafsize-1

        if Q is None:
            #Perform All Nearest Neighbor Search and Compare
            result = self.all_nearest_neighbor(k)

            first = self.forestlist[0]
            truth = first.exact_all_nearest_neighbors(k)
        else:
            #Perform query with the N, d dimensional query points in Q (N x d)
            result = self.aknn(Q, k)
            truth = self.knn(Q, k)

        return Primitives.neighbor_dist(truth, result)


    def clear():
        if self.built:
            self.forestlist = []
        self.data = np.asarray([])
        self.ntrees = 0
        self.size = 0
        self.empty = True


