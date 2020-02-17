
import cupy as cp
from numba import cuda
import prknn.kernels.gpu.core as gpu

@cuda.jit
def add(x, y, out):
    start = cuda.grid(1)
    stride = cuda.gridsize(1)
    for i in range(start, x.shape[0], stride):
        out[i] = x[i] + y[i]

N = 10
#a = cp.ones(N, dtype=cp.float32)
#b = cp.ones(N, dtype=cp.float32) + 1
a = cp.ones(N)
b = cp.ones(N)
out = cp.zeros_like(a)
#c = gpu.add_vectors(a, b)

print(a)
print(b)
print(out)

add[1, 32](a, b, out)
print(out)

#a = cp.arange(N, dtype=cp.float32)
#gpu.numba_increment[1, 32](a)
#print(a)

#a = gpu.sort(a)

#print(a)


