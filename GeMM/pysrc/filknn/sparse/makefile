
INC_DIR=-I${PWD} -I$(TACC_CUDA_DIR)/samples/common/inc

all: sparse

sparse: 
	nvcc -G -g -shared -c ${INC_DIR} sfiknn_test.cu -Xcompiler -fPIC
	mv sfiknn_test.o sfiknn.o

clean:
	rm -f *.o
