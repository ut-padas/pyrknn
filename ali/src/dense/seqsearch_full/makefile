
INC_DIR=-I${TACC_CUDA_DIR}/samples/common/inc -I../norm -I../merge -I../seqsearch_full

all: dense_seq

dense_seq:
	nvcc -shared -c ${INC_DIR} ../norm/Norms.cu ../merge/merge.cu queryknn_seqsearch.cu -Xcompiler -fPIC

clean:


