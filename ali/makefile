


all: dense

sparse_fused:
	cd src/sparse/fused_single_perm && rm -f *.o && rm -f *.so && rm -f *.cpp &&  make
	cd src/sparse && python setup_fused_single_perm.py build_ext --inplace
  
sparse_full:
	cd src/sparse/full_single_perm && rm -f *.o && rm -f *.so && rm -f *.cpp &&  make
	cd src/sparse && python setup_full_single_perm.py build_ext --inplace

dense:
	cd src/dense/full_single_perm && rm -f *.o && rm -f *.so && rm -f *.cpp && $(MAKE)
	cd src/dense && python setup_full_single_perm.py build_ext --inplace
	


