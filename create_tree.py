from prknn.kdforest.mpi.tree import *
from prknn.kdforest.mpi.util import *
from prknn.kdforest.mpi.forest import *


import numpy as np
#import cupy as cp

import time

location = "CPU"

if location == "CPU":
    lib = np
elif location == "GPU":
    lib = cp

built_t = time.time()

N = 16
d = 1
array = lib.random.rand(N, d)

print("Before")
print("-------")
print(array)
#RKDT.set_verbose(True)
tree = RKDT(pointset=array, levels=14, leafsize=4, location=location)
tree.build()

result = tree.ordered_data()

#Print leaf nodes
#for node in tree.get_level(tree.levels):
#    print(node)

#Print new data array
print("After")
print("-------")
print(result)

print(result[tree.gids])

print(array[tree.gids])


