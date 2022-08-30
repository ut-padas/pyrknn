#!/bin/bash

N=$((2**21))
D=16
K=32
ITER=30
LEVELS=11


python dense_rnd.py -n $N -d $D -iter $ITER -levels $LEVELS -k $K


