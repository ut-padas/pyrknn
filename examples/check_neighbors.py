
#Warning: Importing everything into namespace for testing
from knn.forest import *
from knn.tree import *
from knn.util import *

import numpy as np

def test_forest():
    N = 200000
    d = 10
    k = 64
    l = 20

    levels = 12
    leafsize = 512

    assert(k < leafsize)

    np.random.seed(10)          #Fix random seed to debug
    arr = np.random.rand(N, d)  #Generate pointset

    idx = [1, 2, 3, 4, 5]       #Points to query
    q = arr[idx, ...].reshape((len(idx), d))

    print("Building RKDForest...")
    print("N Trees = ", l)
    forest = RKDForest(ntrees=l, levels=levels, leafsize=leafsize, pointset=arr)
    forest.build()
    print("Finished building RKDForest...")

    print("Performing kNN Search..")
    print("(N, d)=", (N, d)," |q|=", len(idx), " leafsize=", leafsize, " k=", k)

    error = forest.check_accuracy(Q=q, k=k)

    print("Recall: ", 100 - error[0]*100)


test_forest()
