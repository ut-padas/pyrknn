#!/bin/bash
#file: set_env.sh

#-- Configure Modules (TACC)
module purge
module load gcc
#module load openmpi
module load spectrum_mpi
module load cuda

export CC="gcc"
export CXX="g++"

export CPLUS_INCLUDE_PATH=$CONDA_PREFIX/include/:$CPLUS_INCLUDE_PATH
export CUDA_ARCH="70"
export PYRKNN_USE_CUDA=1
export MY_SPECTRUM_OPTIONS="--mca common_pami_max_threads 1000"
