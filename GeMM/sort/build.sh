#!/bin/bash


rm -f compare/*.so
rm -f compare/*.o

(cd compare && make sort)

python setup_compare.py build_ext --inplace
python test_sort.py
