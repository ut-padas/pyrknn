from cuda_wrapper.core import add_vectors
import numpy as np

a = np.asarray([1, 1, 1, 1])
b = np.asarray([2, 2, 2, 2])

c = add_vectors(a, b)
print(c)

