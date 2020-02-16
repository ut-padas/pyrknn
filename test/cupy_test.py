
import cupy as cp

import prknn.kernels.gpu.core as gpu

N = 10
a = cp.ones(N, dtype=cp.float32)
b = cp.ones(N, dtype=cp.float32) + 1

c = gpu.add_vectors(a, b)

print(c)


