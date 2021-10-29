#!/bin/bash


(rm -f *.o && rm -f *.so && rm -f *.cpp &&  make)
python setup.py build_ext --inplace 


