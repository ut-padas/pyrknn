

.PHONY: prknn

LIB_UTIL=libutil.so
LIB_KNN=libknngpu.so

GPU_IMPL_DIR=prknn/kernels/gpu/impl

CFLAG=-03 -Wall -I $(EIGEN_ROOT)
VFLAG=-O3 -gencode arch=compute_70,code=sm_70 -shared -Xcompiler -O3,-fPIC
LFLAG=-L$(TACC_CUDA_LIB) -L$(TACC_CUDA_LIB)/stubs/ -lcuda -cudaart -lcublas \
		-L$(GPU_IMPL_DIR)/chao/sort/ -lsortgpu -L.$(GPU_IMPL_DIR)/chao/merge/ -lmergegpu -L$(GPU_IMPL_DIR)/chao/util/ -lutil -L$(GPU_IMPL_DIR)/chao/ -lknngpu


target: prknn $(GPU_IMPL_DIR)/obj/primitives.o $(GPU_IMPL_DIR)/chao/kernel_knn.o $(GPU_IMPL_DIR)/chao/merge/kernel_merge.o
	python setup.py --gpu_obj $(GPU_IMPL_DIR)/obj/primitives.o --setup build_ext --inplace

prknn/kernels/gpu/impl/obj/primitives.o:
	nvcc -shared -c $(GPU_IMPL_DIR)/primitives.cu -o $(GPU_IMPL_DIR)/obj/primitives.o --expt-extended-lambda -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored -Xcompiler -O3,-fPIC

prknn:
	rsync -rupE --delete src/prknn/. prknn

clean:
	rm -rf prknn
