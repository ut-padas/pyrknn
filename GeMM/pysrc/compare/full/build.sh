#!/bin/bash


#(cd dense && rm -f *.o && rm -f *.so && make)
(cd src && rm -f *.o && rm -f *.so && make)


python setup.py build_ext --inplace 
#python setup1.py build_ext --inplace 

