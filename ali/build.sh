#!/bin/bash





cd src/sparse
(cd fused_single_perm && rm -f *.o && rm -f *.so && rm -f *.cpp &&  make)
: '
(cd full_single_perm && rm -f *.o && rm -f *.so && rm -f *.cpp &&  make)
(cd seqsearch_fused && rm -f *.o && rm -f *.so && rm -f *.cpp &&  make)
(cd guided_full && rm -f *.o && rm -f *.so && rm -f *.cpp &&  make)
#(cd guided_full_copydata && rm -f *.o && rm -f *.so && rm -f *.cpp &&  make)
#(cd guided_full_nodatacopy && rm -f *.o && rm -f *.so && rm -f *.cpp &&  make)



python setup_s_full.py build_ext --inplace 
python setup_full_single_perm.py build_ext --inplace 
'
python setup_fused_single_perm.py build_ext --inplace 

#python setup_g_f.py build_ext --inplace 
#python setup_g_f_nc.py build_ext --inplace 
#python setup_g_f_cd.py build_ext --inplace 
cd ../../

: '
cd src/dense 

(cd seqsearch_full && rm -f *.o && rm -f *.so && rm -f *.cpp &&  make)
python setup_s_full.py build_ext --inplace 

cd ../../
'

