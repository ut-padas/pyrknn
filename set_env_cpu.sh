#!/bin/bash
#file: set_env_cpu.sh

module load intel/18
module unload python2/2.7.15

export CUDA_LIB=$TACC_CUDA_LIB
export CUDA_HOME=$TACC_CUDA_LIB

export GSKNN_DIR=$STOCKYARD/maverick2/rnn/
export CUB_ROOT=$STOCKYARD/maverick2/cub/
export MGPU_ROOT=$STOCKYARD/maverick2/moderngpu/src/
export EIGEN_ROOT=$STOCKYARD/maverick2/eigen

export PYRKNN_USE_CUDA=0
export PROD=1

