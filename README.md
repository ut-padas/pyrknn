## PYRKNN: A Distributed Randomized Projection Forest for KNN Graph Construction. 

In PyRKNN we provide support for All-Nearest-Neighbor searches for 32-bit floating point data both dense and sparse in CSR format. 




Installation Notes:
--
The Python interface requires mpi4py, numpy, cupy, scipy, and numba. 
Sklearn is required to run the examples and tests. 



At the moment, the code must be built with either full GPU & CPU support or only CPU support. 

CPU Support requires: GSKNN, MKL, and Eigen. 
GPU Support requires <= CUDA/10.1 (to build the Sparse Kernels), MODERN_GPU

To build with only CPU support set the PYRKNN_USE_CUDA cuda flag to 0. 

Directories to these dependencies must be set. See set_env.sh and set_env_cpu.sh for examples. 


Additional Notes:
----

You might need to set: NUMBA_CUDA_DRIVER=/usr/lib64/libcuda.so.1 if running on Frontera. 





