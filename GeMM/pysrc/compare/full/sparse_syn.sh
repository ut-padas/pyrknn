#!/bin/bash

#N=$((2**22))





#DATASET=random
DATASET=syn
# If the dataset is random
N=$((2**22))
NNZPT=32
D=10000
K=64
ITER=250
LEVELS=12
NQ=1024


python run_sparse_syn.py -levels $LEVELS -dataset $DATASET -n $N -d $D -avgnnz $NNZPT -k $K -nq $NQ 




