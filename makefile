

.PHONY: prknn


target: prknn prknn/kernels/gpu/impl/obj/primitives.o
	python setup.py --gpu_obj prknn/kernels/gpu/impl/obj/primitives.o --setup build_ext --inplace

prknn/kernels/gpu/impl/obj/primitives.o:
	nvcc -shared -c prknn/kernels/gpu/impl/primitives.cu -o prknn/kernels/gpu/impl/obj/primitives.o --expt-extended-lambda -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored -Xcompiler -fPIC

prknn:
	rsync -rupE --delete src/prknn/. prknn

clean:
	rm -rf prknn
