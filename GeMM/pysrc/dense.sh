#!/bin/bash

# gaussian, uniform, sift
DATASET=sift
N=$((2**22))
D=15
K=64
ITER=35
LEVELS=13


python run_dense.py -n $N -d $D -iter $ITER -levels $LEVELS -k $K -dataset $DATASET


