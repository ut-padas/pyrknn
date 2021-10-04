#!/bin/bash



N=$((2**22))


DATASET=sift
D=15
K=64
ITER=70
LEVELS=10
NQ=1024
python -u run_dense.py -n $N -d $D -iter $ITER -levels $LEVELS -k $K -dataset $DATASET -nq $NQ 

