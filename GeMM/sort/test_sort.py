

import numpy as np 

from compare.sort_gpu import *






l = 16384
k = 1024
n_q = 10000


data = np.random.rand(n_q, l)
a = data[0,:]
a = np.sort(a)

nhbd = sort_gpu(data, n_q, l, k)
print(nhbd[0, :10])
print(a[:10])



