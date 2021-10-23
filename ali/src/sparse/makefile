
INC_DIR=-I${TACC_CUDA_DIR}/samples/common/inc -I../../include/sparse/ -I../../include/sparse/dist

all: sparse

sparse:
	nvcc -shared -c ${INC_DIR} dist/GuidedBinSearch.cu dist/Norms.cu queryknn_guided.cu -Xcompiler -fPIC
	#mv sfiknn_v2_SHNP2.o sfiknn.o

clean:


