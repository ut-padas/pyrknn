from prknn.kdforest.mpi.tree import *
from prknn.kdforest.mpi.util import *
from prknn.kdforest.mpi.forest import *

from mpi4py import MPI

import numpy as np
#import cupy as cp

import time



comm = MPI.COMM_WORLD
N = 500000000
d = 10
array = lib.random.rand(N, d)

print("Order Before")
print("-------")
print(array)


#RKDT.set_verbose(True)
tree = RKDT(pointset=array, levels=14, leafsize=512, location="GPU")

build_t = time.time()
tree.build()
build_t = time.time() - build_t

result = tree.ordered_data()

#Print leaf nodes
#for node in tree.get_level(tree.levels):
#    print(node)

#Print new data array
print("Order After")
print("-------")
print(result)
print(type(result))
print("build_t", build_t)
