
INC_DIR=-I${TACC_CUDA_DIR}/samples/common/inc

all: sparse_seq

sparse_seq:
	nvcc -shared -c ${INC_DIR} dist_full_single.cu norm.cu merge.cu queryleafknn.cu -Xcompiler -fPIC

clean:


