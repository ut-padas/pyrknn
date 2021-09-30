
all: simple

simple: simple_cuda.o
	g++ -o program -L$(TACC_CUDA_LIB) -L$(TACC_CUDA_LIB)/stubs/ -lcuda -lcudart simple.cpp simple_cuda.o

simple_cuda.o:
	nvcc -shared -c -I$(TACC_CUDA_LIB)/samples/common/inc ../../SpGeMM_TriPart_v1.cu -Xcompiler -fPIC

clean: 
	rm -f *.o simple
