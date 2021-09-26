
INC_DIR=-I${PWD} -I$(TACC_CUDA_DIR)/samples/common/inc

all: dense

dense: 
	#nvcc -shared -c ${INC_DIR} dfiknn_vcheck3.cu -Xcompiler -fPIC -lcublas
	nvcc -shared -c ${INC_DIR} dfiknn.cu -Xcompiler -fPIC -lcublas
	#mv dfiknn_vcheck3.o dfiknn.o

clean:
	rm -f *.o
