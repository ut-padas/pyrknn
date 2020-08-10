#!/bin/bash
#file: set_env.sh


export CUDA_LIB=$TACC_CUDA_LIB
export CUDA_HOME=$TACC_CUDA_LIB

export GSKNN_DIR=$HOME/rnn/
export CUB_ROOT=$HOME/cub/
export MGPU_ROOT=$HOME/moderngpu/src/
export EIGEN_ROOT=$HOME/eigen/

export PYRKNN_USE_CUDA=1
export PROD=1


