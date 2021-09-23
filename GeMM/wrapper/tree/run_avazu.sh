#!/bin/bash


n=$((2**24))
#leafsize=$((2**11))
levels=17
K=4
python avazu.py -use_gpu 1 -n $n -levels $levels -k $K 



