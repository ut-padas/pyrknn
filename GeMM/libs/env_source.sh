#!/bin/bash

export CUPLA_ROOT=/work2/07544/ghafouri/frontera/gits/pyrknn/GeMM/libs/cupla
export CMAKE_PREFIX_PATH=$CUPLA_ROOT:$TACC_CMAKE_DIR
module load cuda/11.0
module load cmake/3.20.3
#module load gcc/9.1.0
module load intel/18.0.5
module load boost/1.69

