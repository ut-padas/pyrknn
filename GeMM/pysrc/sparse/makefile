
INC_DIR=-I${PWD} -I$(TACC_CUDA_DIR)/samples/common/inc

all: sparse

sparse: 
	nvcc -shared -c ${INC_DIR} sfiknn.cu -Xcompiler -fPIC


clean:
	rm -f *.o
