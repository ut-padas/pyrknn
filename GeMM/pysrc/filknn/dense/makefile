
INC_DIR=-I${PWD} -I$(TACC_CUDA_DIR)/samples/common/inc

all: dense

dense: 
	nvcc -shared -c ${INC_DIR} dfiknn_test.cu -Xcompiler -fPIC -lcublas
	mv dfiknn_test.o dfiknn.o

clean:
	rm -f *.o
