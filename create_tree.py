from prknn.kdforest.merge.tree import *
from prknn.kdforest.merge.util import *
from prknn.kdforest.merge.forest import *


import numpy as np
#import cupy as cp

import time

location = "CPU"

if location == "CPU":
    lib = np
elif location == "GPU":
    lib = cp

built_t = time.time()

N = 10000
d = 5
array = lib.random.rand(N, d)

print("Before")
print("-------")
print(array)
#RKDT.set_verbose(True)
tree = RKDT(pointset=array, levels=14, leafsize=512, location=location)
tree.build()

result = tree.ordered_data()

#Print leaf nodes
#for node in tree.get_level(tree.levels):
#    print(node)

#Print new data array
print("After")
print("-------")
print(result)


