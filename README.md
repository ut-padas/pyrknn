
To build:
- Change paths to dependencies in set_env.sh
- Run make 

Current build dependencies are: 
Libraries: GSKNN, CUB, ModernGPU, Eigen
Python: mpi4py, numpy, numba, cupy, scipy, sklearn, cython

Main functions to perform an all-to-all nearest neighbor search are:
- RKDForest.all_search()
- RKDForest.overlap_search() 



Installation Notes:
--

Numba works on Frontera with CUDA/10.0

Set: CUDA_HOME=$TACC_CUDA_DIR

You might need to set: NUMBA_CUDA_DRIVER=/usr/lib64/libcuda.so.1
