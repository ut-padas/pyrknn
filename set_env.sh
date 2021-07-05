#!/bin/bash
#file: set_env.sh

#-- Configure Modules (TACC)

module load intel/18


#-- Configure GSKNN

#------ Set System Architecture
export GSKNN_DIR=$PWD/extern/gsknn
export GSKNN_ARCH_MAJOR=x86_64
export GSKNN_ARCH_MINOR=sandybridge

export GSKNN_ARCH=$GSKNN_ARCH_MAJOR/$GSKNN_ARCH_MINOR

#------ Set Parallel Options

export KMP_AFFINITY=compact
export OMP_NUM_THREADS=2
export GSKNN_IC_NT=2

#------ Set BLAS Hints

#unused
export GSKNN_USE_BLAS="True"
export MKLROOT=""


#-- Configure PYRKNN

export CUDA_LIB=$TACC_CUDA_LIB
export CUDA_HOME=$TACC_CUDA_LIB
export GSKNN_DIR=$STOCKYARD/maverick2/rnn/
export CUB_ROOT=$STOCKYARD/maverick2/cub/
export MGPU_ROOT=$STOCKYARD/maverick2/moderngpu/src/
export EIGEN_ROOT=$STOCKYARD/maverick2/eigen

export DEBUG=1
export PYRKNN_USE_CUDA=1
export PROD=1

