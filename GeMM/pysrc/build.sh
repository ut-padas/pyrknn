#!/bin/bash


(cd dense && rm -f *.o && rm -f *.so && make)
(cd sparse && rm -f *.o && rm -f *.so && make)

python setup.py build_ext --inplace 

