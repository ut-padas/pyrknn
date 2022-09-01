
INC_DIR=-I${PWD} -I$(TACC_CUDA_DIR)/samples/common/inc

all: sparse

sparse: 
	nvcc -shared -c ${INC_DIR} sfiknn_test_new.cu -Xcompiler -fPIC
	mv sfiknn_test_new.o sfiknn.o

clean:
	rm -f *.o
