from prknn.kdforest.reference.tree import *
from prknn.kdforest.reference.util import *
from prknn.kernels.cpu.core import *

import numpy as np
import time

#This is a large collection of helper function I wrote to debug and verify the reference implementation
#This will not be maintained but you (or most likely me) may find it useful

def test_node_print():
    arr = np.arange(100)
    x = RKDT(pointset=arr)

    arr2 = np.random.rand(100)
    y = RKDT(pointset=arr2)

    RKDT.set_verbose(True)

    x.test_node()
    y.test_node()

def test_node_split():
    N = 20
    arr = np.random.rand(N, 10)
    RKDT.set_verbose(True)
    tree = RKDT(pointset=arr, levels=5, leafsize=5)
    root = RKDT.Node(tree, idx=0, level=0, size=N, gids=np.arange(N))
    children = root.split()
    print(root)
    for child in children:
        grandkids = child.split()
        print(child)
        for kid in grandkids:
            print(kid)

def test_build():
    N = 101
    arr = np.random.rand(N, 5)
    RKDT.set_verbose(True)
    tree = RKDT(pointset=arr, levels=5, leafsize=5)
    tree.build()

    for i in range(tree.get_levels()):
        level = tree.get_level(i)
        gids = []
        for node in level:
            gids.append(node.gids)
        gids = np.concatenate(gids)
        assert(len(np.unique(gids)) == N)

def test_query():
    N = 20
    d = 5
    idx = 10
    arr = np.random.rand(N, d)*1000
    q = arr[idx, ...]
    RKDT.set_verbose(True)
    tree = RKDT(pointset=arr, levels=5, leafsize=5)
    tree.build()

    for i in range(tree.get_levels()):
        level = tree.get_level(i)
        gids = []
        for node in level:
            gids.append(node.gids)
        gids = np.concatenate(gids)
        assert(len(np.unique(gids)) == N)
    
    idx = tree.single_query(q)
    print(idx)


def test_neighbor_node():
    N = 2000
    d = 10
    k = 4
    idx = 5
    arr = np.random.rand(N, d)
    q = arr[idx, ...].reshape((1, d))
    RKDT.set_verbose(True)
    tree = RKDT(pointset=arr, levels=5, leafsize=50)
    tree.build()

    for i in range(tree.get_levels()):
        level = tree.get_level(i)
        gids = []
        for node in level:
            gids.append(node.gids)
        gids = np.concatenate(gids)
        assert(len(np.unique(gids)) == N)

    idx = tree.single_query(q)
    print("Checking Node:", idx)
    query_node = tree.treelist[idx]
    print("With Gids", query_node.gids)
    print("With Data", query_node.data())
    neighbor_list= query_node.knn(q, k)
    print(neighbor_list)

    root = tree.treelist[0]
    true_neighbors = root.knn(q, k)
    print(true_neighbors)


def test_direct():
    N = 20000000
    d = 10
    k = 4
    idx = [1, 2, 3, 4, 5]
    arr = np.random.rand(N, d)
    q = arr[idx, ...].reshape((len(idx), d))
    RKDT.set_verbose(True)
    tree = RKDT(pointset=arr, levels=7, leafsize=5)
    tree.build()

    neighbors = tree.knn(q, k)
    print(neighbors)



def test_neighbor():
    N = 20000000
    d = 10
    k = 4
    idx = [1, 2, 3, 4, 5]
    arr = np.random.rand(N, d)
    q = arr[idx, ...].reshape((len(idx), d))
    RKDT.set_verbose(True)
    tree = RKDT(pointset=arr, levels=7, leafsize=5)
    tree.build()

    neighbors = tree.knn(q, k)
    print(neighbors)

    neighbors = tree.aknn(q, k)
    print(neighbors)



def test_merge():
    N = 20
    d = 10
    k = 4
    idx = [1, 2, 3, 4, 5]
    np.random.seed(10)
    arr = np.random.rand(N, d)
    q = arr[idx, ...].reshape((len(idx), d))
    RKDT.set_verbose(True)
    treeA = RKDT(pointset=arr, levels=7, leafsize=5)
    treeA.build()

    treeB = RKDT(pointset=arr, levels=7, leafsize=5)
    treeB.build()

    print("True Neighbors")
    neighbors = treeA.knn(q, k)
    print(neighbors)

    print("Approximate Neighbors A")
    Aneighbors = treeA.aknn(q, k)
    print(Aneighbors)

    print("Approximate Neighbors B")
    Bneighbors = treeB.aknn(q, k)
    print(Bneighbors)

    Cneighbors = merge_neighbors(Aneighbors, Bneighbors, k)

    print("Merged Neighbors")
    print(Cneighbors)

def test_converge():
    N = 200000
    d = 10
    k = 64
    idx = [1, 2, 3, 4, 5]
    np.random.seed(10)
    arr = np.random.rand(N, d)
    q = arr[idx, ...].reshape((len(idx), d))
    RKDT.set_verbose(True)

    tree = RKDT(pointset=arr, levels=0)
    tree.build()
    tneighbors = tree.knn(q, k)

    lmax = 100
    change_arr = np.zeros(lmax)
    result = None
    for l in range(lmax):
        tree = RKDT(pointset=arr, levels=12, leafsize=512)
        tree.build()

        neighbors = tree.aknn(q, k)

        changes = 0
        if result is None:
            result = neighbors
            change_arr[l] = 0
        else:
            result, changes = merge_neighbors(result, neighbors, k)
            change_arr[l] = changes

        print("NChanges: ", changes)

    print("True Neighbors")
    print(tneighbors)

    print("Merged Neighbors")
    print(result)

    print(neighbor_dist(tneighbors, result))

    print(change_arr)

def test_all_nearest():
    N = 200000
    d = 1
    k = 64
    idx = [1, 2, 3, 4, 5]
    arr = np.random.rand(N, d)
    q = arr[idx, ...].reshape((len(idx), d))
    RKDT.set_verbose(True)
    tree = RKDT(pointset=arr, levels=15, leafsize=512)
    tree.build()

    #neighbors = tree.aknn(q, k)
    #print(neighbors)

    neighbors = tree.all_nearest_neighbor(k);
    print(neighbors)

def test_distance():
    N = 1
    d = 5
    a = np.arange(N)
    a = a.reshape(N, 1)
    y = np.ones(d)
    a = a + y
    print(a)
    b = np.zeros([2, d])
    b[1] = 10
    print(b)
    D = distance(a, b)
    print(D)
    dist = D[0] - D[1]
    print(dist)
    if dist < 5:
        print("True")

def test_neighbor_kernel():
    N = 100
    d = 5
    R = np.random.rand(N, d);
    R = np.asarray(R, dtype=np.float32)
    k = 15
 
    idx = np.asarray([0, 1, 2], dtype=np.int32)
    Q = R[idx, ...].reshape((len(idx), d))

    gids = np.arange(0, N, dtype=np.int32);   

    #print(direct_knn(gids, R, Q, k))
    #a = PyGSKNN(gids, R, Q, k)
    b = KNNLowMem(gids, R, Q, k)
    c = PyGSKNNBlocked(gids, R, Q, k)
    #print(b)
    #print(a)
    print(c)

    print(np.sort(b[0][1, :]))
    print(np.sort(c[0][1, :]))

def test_neighbor_multileaf():
    #test_par()
    numpy_time = []
    stl_time = []
    gsknn_time = []
    block_time = []
    batch_time = []
    MAX = 20;
    MT = 2;
    for size in range(5, MAX, 5):
        leaves = size;
        N = 2048
        d = 5
        k = 64
        Rlist = []
        Qlist = []
        gidsList = []
        AnsList = []
        print(leaves)
        for l in range(leaves):
            R = np.random.rand(N, d);
            R = np.asarray(R, dtype=np.float32);
            #idx = np.asarray([0, 1, 2], dtype=np.int32)
            #Q = R[idx, ...].reshape((len(idx), d))
            Q = R
            gids = np.arange(0, N, dtype=np.int32);   

            Rlist.append(R)
            Qlist.append(Q)
            gidsList.append(gids)

        trial = []
        for t in range(MT):
            AnsList = []
            start_t = time.time()
                
            for l in range(leaves):
                direct_knn(gids, Rlist[l], Qlist[l], k)
            t_NUMPY = time.time() - start_t
            print("NUMPY:", t_NUMPY)
            trial.append(t_NUMPY)
        t_NUMPY = np.mean(trial)
        
        """
        trial = []
        for t in range(MT):
            AnsList = []
            t_STL = time.time()
            for l in range(leaves):
                KNNLowMem(gids, Rlist[l], Qlist[l], k)
            t_STL = time.time() - t_STL
            print("LOWMEM:", t_STL)
            trial.append(t_STL)
        t_STL = np.mean(trial)
        """
        """
        trial = []
        for t in range(MT):
            AnsList = []
            t_GSKNN = time.time()
            for l in range(leaves):
                AnsList.append(PyGSKNN(gids, Rlist[l], Qlist[l], k)[0])
            t_GSKNN = time.time() - t_GSKNN
            print("GSKNN:", t_GSKNN)
            trial.append(t_GSKNN)
        t_GSKNN = np.mean(trial)
        """
        """
        trial = []
        for t in range(MT):
            AnsList = []
            t_GSKNN_block = time.time()
            for l in range(leaves):
                PyGSKNNBlocked(gids, Rlist[l], Qlist[l], k)
            t_GSKNN_block = time.time() - t_GSKNN_block
            print("BLOCKED:", t_GSKNN_block)
            trial.append(t_GSKNN_block)
        t_GSKNN_block = np.mean(trial)
        """
        
        """
        trial = []
        for t in range(MT):
            t_batchedGSKNN = time.time()
            a = PyGSKNNBatched(gidsList, Rlist, Qlist, k)
            t_batchedGSKNN = time.time() - t_batchedGSKNN
            trial.append(t_batchedGSKNN)
            print("BATCH:", t_batchedGSKNN)
        t_batchedGSKNN = np.mean(trial)
        """

        numpy_time.append(t_NUMPY)
        #stl_time.append(t_STL)
        #gsknn_time.append(t_GSKNN)
        #block_time.append(t_GSKNN_block)
        #batch_time.append(t_batchedGSKNN)

    print(numpy_time)
    #print(stl_time)
    #print(gsknn_time)
    ##print(block_time)
    #print(batch_time)
    #print(a)
    #print(AnsList)
    
def time_full_equivalent():
        N = 2048
        d = 100
        k = 64
        MAX = 100
        MT = 10;
        full_time = []
        for leaves in range(5, MAX, 5):
            print(leaves)
            Q = np.random.rand(leaves*N, d);
            Q = np.asarray(Q, dtype=np.float32);
            idx = np.asarray(np.arange(N), dtype=np.int32)
            R = Q[idx, ...].reshape((len(idx), d))
            gids = np.arange(0, leaves*N, dtype=np.int32);   
            
            trial = []
            for t in range(MT):
                AnsList = []
                t_GSKNN = time.time()
                PyGSKNN(gids, R, Q, k)
                t_GSKNN = time.time() - t_GSKNN
                print("FULL:", t_GSKNN)
                trial.append(t_GSKNN)
            t_GSKNN = np.mean(trial)
            full_time.append(t_GSKNN)

        print(full_time)


def time_numpy():
        N = 2048
        d = 10
        k = 64
        MAX = 100
        MT = 10;
        full_time = []
        for leaves in range(5, MAX, 5):
            print(leaves)
            R = np.random.rand(leaves*N, d);
            R = np.asarray(R, dtype=np.float32);
            idx = np.asarray(np.arange(N), dtype=np.int32)
            Q = R[idx, ...].reshape((len(idx), d))
            gids = np.arange(0, leaves*N, dtype=np.int32);   
            
            trial = []
            for t in range(MT):
                AnsList = []
                L = R.T
                t_GSKNN = time.time()
                A = Q @ L
                t_GSKNN = time.time() - t_GSKNN
                print("FULL:", t_GSKNN)
                print(A.shape)
                trial.append(t_GSKNN)
            t_GSKNN = np.mean(trial)
            full_time.append(t_GSKNN)

        print(full_time)

def test_multileaf():
    #test_par()
    leaves = 20;
    N = 2560
    d = 5
    k = 15
    Rlist = []
    Qlist = []
    gidsList = []
    AnsList = []
    for l in range(leaves):
        R = np.random.rand(N, d);
        R = np.asarray(R, dtype=np.float32);
        idx = np.arange(256, dtype=np.int32)
        #idx = np.asarray([0, 1, 2], dtype=np.int32)
        Q = R[idx, ...].reshape((len(idx), d))
        #Q = R
        gids = np.arange(0, N, dtype=np.int32);   

        Rlist.append(R)
        Qlist.append(Q)
        gidsList.append(gids)

    t_batchedGSKNN = time.time()
    a = PyGSKNNBatched(gidsList, Rlist, Qlist, k)
    t_batchedGSKNN = time.time() - t_batchedGSKNN

    AnsList = []
    t_GSKNN = time.time()
    for l in range(leaves):
        AnsList.append(PyGSKNN(gids, Rlist[l], Qlist[l], k)[0])
    t_GSKNN = time.time() - t_GSKNN

    print(a)
    print(AnsList[:][:])
    print("END")

#test_distance()
#test_node_split()
#test_build()
#test_query()

#test_neighbor_kernel()
test_neighbor_multileaf()
#time_full_equivalent()
#time_numpy()
#test_multileaf()
#test_neighbor_node()
#test_neighbor()

#test_all_nearest()
#test_merge()

#test_converge()

