from pyrknn.kdforest.hash_bad.tree import *
from pyrknn.kdforest.hash_bad.util import *
from pyrknn.kdforest.hash_bad.forest import *


import numpy as np
import os
from sklearn.preprocessing import normalize
from sklearn.datasets import fetch_openml
from mlxtend.data import loadlocal_mnist
import pandas as pd 

N = 2**21
d = 20
k  = 64
nq = 1000

fpath = os.environ["SCRATCH"]+"/datasets/mnist/infimnist/infimnist/data/"
#data = np.random.rand(N, d)
#data = np.load(os.environ["SCRATCH"]+"/datasets/mnist/mnist_small.npy")
#data = np.asarray(data>1, dtype=np.float32)
#data = normalize(data)
#mnist = fetch_openml('mnist_784', version=1, cache=True, as_frame=False)
#data = mnist.data
#data = np.asarray(data, dtype=np.float32)
#data = normalize(data)
X, y = loadlocal_mnist(
        images_path=fpath+'train-images-idx3-ubyte', 
        labels_path=fpath+'train-labels-idx1-ubyte')
data = X
data = np.load(os.environ["SCRATCH"]+"/datasets/sphere/sphere_1.npy")

df = pd.read_csv('/scratch1/06081/wlruys/datasets/glove/glove.6B.50d.txt', sep=" ", quoting=3, header=None, index_col=0)
data = df.to_numpy()

data = np.asarray(data, dtype=np.float32)
data = data[:2**19, :200]
print(data.shape)

Q = np.copy(data[:nq, :d])

forest = RKDForest(data, leafsize=256)

truth = forest.direct(Q, k)
print(truth)

approx = forest.search(k, truth=truth, until=True, threshold=0.98, cores=56, until_max=100)
#print(approx)
#print(truth)
#print(truth[0]-approx[0])
#print(neighbor_dist(truth, truth))
record = util.Recorder()
record.write("acc.csv")
"""
print(data)
X = np.copy(data)
tree = RKDT(None, X, leafsize=N//2)
tree.build()
print(X)
ids = np.arange(N)
print(X[ids[tree.gids]])
"""
