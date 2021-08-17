#!/bin/bash


rm -f cuda_wrapper/*.so

(cd cuda_wrapper && make)

python setup.py build_ext --inplace
python test_gemm.py

