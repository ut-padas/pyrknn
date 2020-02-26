import cupy as cp
import numpy as np
import time
import csv
from util import *
from tree_gpu_cpu import *

d = 100

csv_path = "time_build_tree_d"+str(d)+".csv"
csv_file = open(csv_path, 'w')
csv_writer = csv.writer(
            csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

csv_writer.writerow(["size", "time"])
libpy = cp
for size in range(1000000, 10000001, 1000000):
    cp.random.RandomState(1001)
    data_set = cp.random.random((size,d),dtype='float32')
    t0 = time.time()
    tree = RKDT(libpy, pointset=data_set, levels=5, leafsize=1024)
    tree.build()
    t1 = time.time()
    del data_set
    csv_writer.writerow([size,t1-t0])
    csv_file.flush()
