#!/bin/bash



rm -rf cuda_wrapper/*.so
rm -rf cuda_wrapper/*.o



(cd cuda_wrapper && make dense) 


python setup_dense.py build_ext --inplace


python test_dgemm.py


