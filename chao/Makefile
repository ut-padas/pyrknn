
CC=g++
CFLAG=-O3 -Wall -I $(EIGEN_ROOT)
VFLAG=-O3 -DTHRUST_DEBUG -arch=compute_70 -code=sm_70 -shared -Xcompiler -fPIC
LFLAG=-L$(TACC_CUDA_LIB) -L$(TACC_CUDA_LIB)/stubs/ -lcuda -lcudart -lcublas

EXE=$(SCRATCH)/bin/knn.exe

default: $(EXE)

$(EXE): knn.o kernel_gpu.o timer.o
	nvcc knn.o kernel_gpu.o timer.o -o $(EXE) $(LFLAG)
	
knn.o: knn.cpp knn.hpp
	$(CC) -c knn.cpp $(CFLAG)

kernel_gpu.o: kernel_gpu.cu kernel_gpu.hpp
	nvcc -c kernel_gpu.cu $(VFLAG)

timer.o: timer.cpp timer.hpp
	$(CC) -c timer.cpp $(CFLAG)

run: $(EXE)
	$(EXE)

debug: $(EXE)
	$(EXE) -D -n 2 -d 2 -k 2

clean:
	rm -f *.o $(EXE)

