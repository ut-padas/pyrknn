
#INC_DIR=-I${TACC_CUDA_DIR}/samples/common/inc -I../../include/sparse -I../../include/sparse/dist -I../../include/sparse/merge
INC_DIR=-I${TACC_CUDA_DIR}/samples/common/inc -I../../include/sparse -I../../include/sparse/dist -I../../include/sparse/merge

all: sparse_seq

sparse_seq:
	nvcc -shared -c ${INC_DIR} dist/SeqBinSearch.cu dist/Norms.cu merge/merge.cu queryknn_seqsearch.cu -Xcompiler -fPIC
	#nvcc -shared -c ${INC_DIR} dist/SeqBinSearch.cu dist/Norms.cu merge/merge.cu queryknn_seqsearch.cu -Xcompiler -fPIC
	#mv sfiknn_v2_SHNP2.o sfiknn.o

sparse_guided:
	nvcc -shared -c ${INC_DIR} dist/GuidedBinSearch.cu dist/Norms.cu merge/merge.cu queryknn_guided.cu -Xcompiler -fPIC

clean:


