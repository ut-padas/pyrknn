
import cupy as cp
import prknn.kernels.gpu.core as gpu
import time

N = 100001
a = cp.arange(N, dtype=cp.float32)
a = cp.random.rand(N, dtype=cp.float32)
print(a)
t = time.time()
out = gpu.median(a)
elapsed = time.time() -t
print(elapsed)
print(out)

s = cp.sort(a)
print(s[N/2])
print(s[N/4])



