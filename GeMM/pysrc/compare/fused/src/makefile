
INC_DIR=-I${PWD} -I$(TACC_CUDA_DIR)/samples/common/inc

all: sparse

sparse: 
	#nvcc -shared -c ${INC_DIR} sfiknn_v2_SHNP2.cu -Xcompiler -fPIC
	nvcc -shared -c ${INC_DIR} SpGeMM_Fused.cu -Xcompiler -fPIC
	mv SpGeMM_Fused.o sfiknn.o
	#mv sfiknn_v2_SHNP2.o sfiknn.o

clean:
	rm -f *.o
