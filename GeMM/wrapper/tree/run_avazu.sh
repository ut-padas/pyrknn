#!/bin/bash


n=$((2**21))
leafsize=$((2**11))
python exurl.py -use_gpu 1 -n $n -leafsize $leafsize



