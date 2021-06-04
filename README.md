## PYRKNN: A Distributed Randomized Projection Forest for KNN Graph Construction. 

In PyRKNN we provide support for All-Nearest-Neighbor searches for 32-bit floating point data both dense and sparse in CSR format. 

The search routine is provided by RKDForest object, where location specifies whether the local search is conducted on the CPU "HOST" or GPU "GPU".  
The routine can be called from within MPI. By default it assumes the global communicator, otherwise subcommunicators can be passed in to the forest constructor. 

```
forest = RKDForest(data=X, leafsize=leafsize, location="HOST", comm=MPI.COMM_WORLD)
neighbors = forest.all_search(k)
```



Installation Notes:
--
The Python interface requires mpi4py, numpy, cupy, scipy, and numba>=0.52. 
Sklearn is required to run the examples and tests. 

At the moment, the code must be built with either full GPU & CPU support or only CPU support. 

CPU Support requires: GSKNN (https://github.com/ChenhanYu/rnn) , MKL, and Eigen (https://gitlab.com/libeigen/eigen).

GPU Support requires <= CUDA/10.1, moderngpu (https://moderngpu.github.io/intro.html) 

To build with only CPU support set the PYRKNN_USE_CUDA=0. 

Directories to these dependencies must be set. See set_env.sh and set_env_cpu.sh for examples. 


Additional Notes:
----

You might need to set: NUMBA_CUDA_DRIVER=/usr/lib64/libcuda.so.1 if running on Frontera. 





