
INC_DIR=-I${PWD} -I$(TACC_CUDA_DIR)/samples/common/inc

all: dense

dense: 
	nvcc -shared -c ${INC_DIR} dfiknn_SHNP2_v4.cu -Xcompiler -fPIC -lcublas
	mv dfiknn_SHNP2_v4.o dfiknn.o

clean:
	rm -f *.o
