
INC_DIR=-I${TACC_CUDA_DIR}/samples/common/inc

all: dense_full_single

dense_full_single:
	nvcc -shared -c ${INC_DIR} norm.cu merge.cu queryleafknn.cu -Xcompiler -fPIC

clean:


