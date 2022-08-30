#!/bin/bash

#N=$((2**22))


DATASET=url
# If the dataset is random
N=2396130
NNZPT=16

D=100
K=64
ITER=150
LEVELS=12


python -u run_sparse.py -n $N -d $D -avgnnz $NNZPT -iter $ITER -levels $LEVELS -k $K -dataset $DATASET


