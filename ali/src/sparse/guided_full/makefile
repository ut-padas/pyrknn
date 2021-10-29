
#INC_DIR=-I${TACC_CUDA_DIR}/samples/common/inc -I../../include/sparse -I../../include/sparse/dist -I../../include/sparse/merge
INC_DIR=-I${TACC_CUDA_DIR}/samples/common/inc -I../merge -I../norm

all: sparse_guided

sparse_guided:
	nvcc -lineinfo -shared -c ${INC_DIR} GuidedBinSearch.cu ../norm/Norms.cu ../merge/merge.cu queryknn_guided.cu -Xcompiler -fPIC

clean:


