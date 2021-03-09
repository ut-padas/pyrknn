from pyrknn.kdforest.mpi.tree import *
from pyrknn.kdforest.mpi.util import *
from pyrknn.kdforest.mpi.forest import *

from mpi4py import MPI

import numpy as np

import time
import platform
import pickle 

from sklearn.datasets import fetch_openml
from sklearn.preprocessing import normalize
from mlxtend.data import loadlocal_mnist

import pandas as pd 

weak = False
sparse = False

def build():
    #Build all types of treehash with different depths of trees
    for X in [2, 4, 8, 16, 32, 64, 128, 256]:

        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()

        f = 'uniform/uniform_'
        f = 'sphere/sphere_'
        f = 'hard/hard_'

        np.random.seed(1001)

        d = 15
        N = 2**15
        data = None
        #filename = os.environ["SCRATCH"]+'/datasets/'+f+str(rank)+'.npy'

        #tdata = np.load(filename)
        #print(tdata.shape)
        #tdata = np.asarray(tdata, dtype=np.float32)
        #if data is None:
        #    data = tdata
        #data = data[:N, :]

        #data = np.copy(data[:N, :d])
        #fpath = os.environ["SCRATCH"]+"/datasets/mnist/infimnist/infimnist/"
        #X, y = loadlocal_mnist(
        #        images_path=fpath+'mnist8m-patterns-idx3-ubyte', 
        #        labels_path=fpath+'mnist8m-labels-idx1-ubyte')
        #data = X[:60000, :]

        #df = pd.read_csv('/scratch1/06081/wlruys/datasets/glove/glove.6B.50d.txt', sep=" ", quoting=3, header=None, index_col=0)
        #data = df.to_numpy()
        #filename = os.environ["SCRATCH"]+'/datasets/'+f+str(0)+'.npy'
        #tdata = np.load(filename)
        #tdata = np.asarray(tdata, dtype=np.float32)
        
        data = np.random.randn(N, d)
        #data = normalize(data)
        #data = np.asarray(data[:N, :], dtype=np.float32)
        #tdata = data

        #mnist = fetch_openml('higgs', version=1, cache=True, as_frame=False)
        #data = mnist.data

        #data = data + np.random.rand(data.shape[0], data.shape[1])
        #data = np.asarray(data>0, dtype=np.float32)/255
        #data = data - np.mean(data)
        #print(data)

        data = np.asarray(data[:N, :], dtype=np.float32)
        tdata = data

        k = 128
        nq = 100
        #Q = np.copy(tdata[:nq, :d])

        #A = np.copy(data) 
        #tree = RKDForest(pointset=A, levels=0, leafsize=256, location="CPU", ntrees=1, sparse=sparse)
        #tree.build()
        #tree = tree.forestlist[0]
        #truth = tree.dist_exact(Q, k)

        N = min(N, data.shape[0])

        #truth = (np.zeros([nq, k], dtype=np.int32), np.zeros([nq, k], dtype=np.float32))
        B = np.copy(data)
        ls = N//X
        print(",N: ", N, ",d: ", d, ",leafsize: ", ls)
        max_tree = 100
        name  = "gauss15d"
        fname = name+"_"+str(ls)+"_hash.bin"
        dname = name+"_"+str(ls)+"_data.bin"
        tree = RKDForest(pointset=data, levels=20, leafsize=ls, comm=comm, location="CPU", ntrees=max_tree, sparse=sparse)
        approx = tree.aknn_all_build(k, ntrees=1, blockleaf=2, blocksize=32, cores=56)
        global htable 
        np.save(fname, htable[:N, :max_tree])
        np.save(dname, B)


build()
