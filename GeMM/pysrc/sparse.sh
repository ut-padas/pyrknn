#!/bin/bash

N=$((2**23))
D=100
NNZPT=16
K=32
ITER=1
LEVELS=13


python sparse_rnd.py -n $N -d $D -avgnnz $NNZPT -iter $ITER -levels $LEVELS -k $K


