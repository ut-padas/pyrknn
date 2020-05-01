#!/bin/bash
#file: set_env.sh

module load gcc/7.3 openmpi cuda/10.1

export LD_PRELOAD=$TACC_CUDA_LIB/libcusparse.so:$TACC_CUDA_LIB/libcudart.so:/lib64/libcublas.so:$LD_PRELOAD

export CUDA_LIB=$TACC_CUDA_LIB
export CUDA_HOME=$TACC_CUDA_LIB

export GSKNN_DIR=$HOME/maverick2/rnn/
export CUB_ROOT=$HOME/cub/
export MGPU_ROOT=$HOME/moderngpu/src/
export EIGEN_ROOT=$HOME/eigen

export PRKNN_USE_CUDA=1
export PROD=1


