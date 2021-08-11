#!/bin/bash

rm -rf cuda_wrapper

cp -R dev cuda_wrapper

source activate py37

python setup.py build_ext --inplace

conda deactivate
