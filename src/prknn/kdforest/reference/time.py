import cupy as cp
import numpy as np
import time
import csv
from util import *

d = 100
ref_size = 2000

kernel = "knn_stream_kernel1"
csv_path = "time_"+kernel+"d"+str(d)+"refsize"+str(ref_size)+".csv"
csv_file = open(csv_path, 'w')
csv_writer = csv.writer(
            csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

csv_writer.writerow(["query_size", "time"])
k = 10
cp.random.RandomState(1001)
refs = cp.random.random((ref_size,d))
refs_norm_sq = cp.linalg.norm(refs, axis = -1)

for query_size in range(5, 64, 2):
    cp.random.RandomState(1001)
    querys = cp.random.random((query_size,d))
    t0 = time.time()
    globals()[kernel](querys,refs,refs_norm_sq,k)
    t1 = time.time()
    csv_writer.writerow([query_size,t1-t0])
