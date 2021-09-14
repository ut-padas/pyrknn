#!/bin/bash
#file: set_env.sh

#-- Configure Modules (TACC)
module purge
module load gcc
module load openmpi
module load cuda

export CPLUS_INCLUDE_PATH=$CONDA_PREFIX/include/:$CPLUS_INCLUDE_PATH
export CUDA_ARCH="70"
export PYRKNN_USE_CUDA=1

