from prknn.kdforest.merge.tree import *
from prknn.kdforest.merge.util import *
from prknn.kdforest.merge.forest import *


import numpy as np
#import cupy as cp

import time

location = "GPU"

built_t = time.time()

N = 2**23
d = 10
array = cp.random.rand(N, d)

print("Before")
print("-------")
print(array)
#RKDT.set_verbose(True)
tree = RKDT(pointset=array, levels=14, leafsize=4, location=location)

build_t = time.time()
tree.build()
build_t = time.time() - build_t
print("Time:", build_t)

result = tree.ordered_data()

#Print leaf nodes
#for node in tree.get_level(tree.levels):
#    print(node)

#Print new data array
print("After")
print("-------")
print(result)



