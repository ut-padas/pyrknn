
default: gemm

CFLAG=-O3 -arch=compute_70 -code=sm_70 -shared -Xcompiler -fPIC
LFLAG=-L$(TACC_CUDA_LIB) -L$(TACC_CUDA_LIB)/stubs/ -lcuda -lcudart -lcublas

gemm: gemm.o timer.o
	g++ $^ -o $@ $(LFLAG)

gemm.o: gemm.cu
	nvcc -c gemm.cu $(CFLAG) 

timer.o: timer.cpp timer.hpp
	g++ -c timer.cpp -Wall -O3

run: gemm
	./gemm

clean:
	rm -rf *.o gemm


