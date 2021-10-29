
INC_DIR=-I${TACC_CUDA_DIR}/samples/common/inc -I../norm -I../merge

all: sparse_seq

sparse_seq:
	nvcc -shared -c ${INC_DIR} SeqBinSearch.cu ../norm/Norms.cu ../merge/merge.cu queryknn_seqsearch.cu -Xcompiler -fPIC

clean:


