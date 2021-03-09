
import numpy as np

from . import util 
from .tree import *



class RKDForest:

    def __init__(self, data, leafsize=None, label="id", hlength=10):
        self.data = data
        self.N = data.shape[0]
        self.d = data.shape[1]
        self.leafsize = leafsize
        self.gids = np.arange(self.N, dtype=np.int32)

        if label == "id":
            self.hash = np.zeros( (self.N, hlength) )
        elif label == "center":
            self.hash = np.zeros( (self.N, hlength, self.d) )
    
    def direct(self, Q, k):
        results = util.single_knn(self.gids, self.data, Q, k)
        results = util.merge_neighbors(results, results, k)
        return results 

    def search(self, k, ntrees=1, cores=4, truth=None, until=False, until_max=100, gap=5, threshold=0.95):
        util.set_cores(cores)
        record = util.Recorder()
        result = None
        if truth:
            nq = truth[0].shape[0]

        if until:
            self.ntrees = until_max
        
        for it in range(self.ntrees):
            X = np.copy(self.data)
            tree = RKDT(self, X, leafsize=self.leafsize)
            tree.build()

            neighbors = tree.aknn_all(k)

            if result is None:
                result = util.merge_neighbors(neighbors, neighbors, k)
            else:
                result = util.merge_neighbors(result, neighbors, k)
            #print(neighbors)
            #print(result)

            break_flag = False

            rlist, rdist = result
            test = (rlist[:nq], rdist[:nq])
            acc = util.neighbor_dist(truth, test)
            record.push("Recall", acc[0])
            record.push("Distance", acc[1])

            print("Iteration", it, "Recall: ", acc, flush=True)

            if until and acc[0] > threshold:
                break_flag = True

            if break_flag:
                break

        return result
            
