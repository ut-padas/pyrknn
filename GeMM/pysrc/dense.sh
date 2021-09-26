#!/bin/bash

N=$((2**22))
D=15
K=64
ITER=40
LEVELS=13


python dense_rnd.py -n $N -d $D -iter $ITER -levels $LEVELS -k $K


