
To build:
- Change paths to dependencies in set_env.sh
- Run make 

Current build dependencies are: 

Libraries: GSKNN, ModernGPU, Eigen

Python: mpi4py, numpy, numba, cupy, scipy, sklearn, cython

Main functions to perform an all-to-all nearest neighbor search are:
- RKDForest.all_search()
- RKDForest.overlap_search() 



Installation Notes:
--

GPU Sparse Kernel requires <= CUDA/10.1. Several of the calls are deprecated in CUDA 11 and have a new interface. 

Numba works on Frontera with CUDA/10.0

Set: CUDA_HOME=$TACC_CUDA_DIR

You might need to set: NUMBA_CUDA_DRIVER=/usr/lib64/libcuda.so.1


#TODO
**Will**: Run max int32 problem Dense synthetic
**Will**: Strong scaling for largest 1B SIFT and 100M  SIFT runs,  dense CPU
**George/Ali**: Criteo
**George**: scalable correct synthetic sparse
**Chao**: GPU/CPU Dense discrepency, bug
