
import cupy as cp

import prknn.kernels.gpu.core as gpu

N = 100
a = cp.ones(N)
b = cp.ones(N) + 1

c = gpu.add_vectors(a, b)

print(c)


