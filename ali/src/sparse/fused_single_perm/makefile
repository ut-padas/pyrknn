
INC_DIR=-I${TACC_CUDA_DIR}/samples/common/inc 

all: sparse_seq

sparse_seq:
	nvcc -shared ${INC_DIR} -c merge.cu fused_single_perm.cu norm.cu queryleafknn.cu -Xcompiler -fPIC

clean:


