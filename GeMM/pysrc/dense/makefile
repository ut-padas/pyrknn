
INC_DIR=-I${PWD} -I$(TACC_CUDA_DIR)/samples/common/inc

all: dense

dense: 
	nvcc -shared -c ${INC_DIR} dfiknn.cu -Xcompiler -fPIC -lcublas


clean:
	rm -f *.o
