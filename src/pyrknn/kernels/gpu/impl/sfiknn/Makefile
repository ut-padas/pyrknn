

CODE_DIR=/work2/07544/ghafouri/frontera/gits/pyrknn/GeMM
INC_DIR=-I$(CODE_DIR)/include -I$(CUDA_HOME)/samples/common/inc 
DEV_DIR=$(CODE_DIR)/src


all: sparse

sparse:
	nvcc -shared -c ${INC_DIR} $(DEV_DIR)/sfiknn.cu -Xcompiler -fPIC

clean:
	rm -f *.o 

