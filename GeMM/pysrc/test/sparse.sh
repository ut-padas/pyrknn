#!/bin/bash

#N=$((2**22))


DATASET=avazu
# If the dataset is random
N=2396130
NNZPT=16
D=100
K=4
ITER=50
LEVELS=16


python -u run_sparse.py -n $N -d $D -avgnnz $NNZPT -iter $ITER -levels $LEVELS -k $K -dataset $DATASET


