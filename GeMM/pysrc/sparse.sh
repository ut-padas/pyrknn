#!/bin/bash

#N=$((2**22))


DATASET=url
# If the dataset is random
N=2396130
NNZPT=16

D=100
K=64
ITER=1
LEVELS=12


python run_sparse.py -n $N -d $D -avgnnz $NNZPT -iter $ITER -levels $LEVELS -k $K -dataset $DATASET


