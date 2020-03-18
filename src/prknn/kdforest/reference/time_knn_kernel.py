import cupy as cp
import numpy as np
import time
import csv
from util import *

d = 100
ref_size = 2000000

kernel = "cupy_dot_transpose"
csv_path = "time_"+kernel+"d"+str(d)+"refsize"+str(ref_size)+".csv"
csv_file = open(csv_path, 'w')
csv_writer = csv.writer(
            csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

csv_writer.writerow(["query_size", "time"])
k = 10
cp.random.RandomState(1001)
refs = cp.random.random((ref_size,d))
refs_norm_sq = cp.linalg.norm(refs, axis = -1)

for query_size in range(10, 2000, 10):
    cp.random.RandomState(1001)
    querys = cp.random.random((query_size,d))
    t0 = time.time()
    #globals()[kernel](querys,refs,refs_norm_sq,k)
    result = cp.dot(refs,cp.transpose(querys))
    t1 = time.time()
    del result
    csv_writer.writerow([query_size,t1-t0])
    csv_file.flush()
