
INC_DIR=-I${PWD} -I$(TACC_CUDA_DIR)/samples/common/inc

all: dense

dense: 
	#nvcc -shared -c ${INC_DIR} dfiknn_SHNP2_v4.cu -Xcompiler -fPIC -lcublas
	nvcc -shared -c ${INC_DIR} dfiknn_test.cu -Xcompiler -fPIC -lcublas
	#mv dfiknn_SHNP2_v4.o dfiknn.o
	mv dfiknn_test.o dfiknn.o

clean:
	rm -f *.o
