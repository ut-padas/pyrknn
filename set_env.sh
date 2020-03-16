#!/bin/bash
#file: set_env.sh


export CUDA_LIB=$TACC_CUDA_LIB
export CUDA_HOME=$TACC_CUDA_LIB

export GSKNN_DIR=$STOCKYARD/maverick2/rnn/
export CUB_ROOT=$STOCKYARD/maverick2/cub/
export MGPU_ROOT=$STOCKYARD/maverick2/moderngpu/src/
export EIGEN_ROOT=$STOCKYARD/maverick2/

export PRKNN_USE_CUDA=1



