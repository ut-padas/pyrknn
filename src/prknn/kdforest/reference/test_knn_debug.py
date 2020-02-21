from tree_gpu_cpu import *
from util import *
import numpy as np
import cupy as cp

#This is a large collection of helper function I wrote to debug and verify the reference implementation
#This will not be maintained but you (or most likely me) may find it useful
libpy = np


def test_node_print():
    arr = np.arange(100)
    x = RKDT(libpy, pointset=arr)

    arr2 = np.random.rand(100)
    y = RKDT(libpy, pointset=arr2)

    RKDT.set_verbose(True)

    x.test_node()
    y.test_node()

def test_node_split():
    N = 20
    arr = np.random.rand(N, 10)
    RKDT.set_verbose(True)
    tree = RKDT(libpy,pointset=arr, levels=5, leafsize=5)
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
    tree = RKDT(libpy,pointset=arr, levels=5, leafsize=5)
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
    q = arr[idx, ...].reshape((1, d))
    RKDT.set_verbose(True)
    tree = RKDT(libpy,pointset=arr, levels=5, leafsize=5)
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
    tree = RKDT(libpy,pointset=arr, levels=5, leafsize=50)
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
    tree = RKDT(libpy,pointset=arr, levels=7, leafsize=5)
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
    tree = RKDT(libpy, pointset=arr, levels=7, leafsize=5)
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
    treeA = RKDT(libpy, pointset=arr, levels=7, leafsize=5)
    treeA.build()

    treeB = RKDT(libpy, pointset=arr, levels=7, leafsize=5)
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

    tree = RKDT(libpy, pointset=arr, levels=0)
    tree.build()
    tneighbors = tree.knn(q, k)

    lmax = 100
    change_arr = np.zeros(lmax)
    result = None
    for l in range(lmax):
        tree = RKDT(libpy, pointset=arr, levels=12, leafsize=512)
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
    tree = RKDT(libpy, pointset=arr, levels=15, leafsize=512)
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

def test_knn_stream_kernel1():
    querys = cp.array([[0,0],[1,1]], dtype="float32")
    refs = cp.array([[0,0.5], [0.5,0], [1,1.5]])
    results = knn_stream_kernel1(querys, refs, 1)
    print(results)

test_distance()
test_node_split()
test_build()
test_query()
test_neighbor_node()
test_neighbor()
test_knn_stream_kernel1()
test_all_nearest()
test_merge()

test_converge()


