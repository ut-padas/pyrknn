
CC=g++
CFLAG=-O3 -Wall -I $(EIGEN_ROOT)
VFLAG=-O3 -gencode arch=compute_70,code=sm_70 -shared -Xcompiler -fPIC
LFLAG=-L$(TACC_CUDA_LIB) -L$(TACC_CUDA_LIB)/stubs/ -lcuda -lcudart -lcublas \
			-L ./sort/ -lsortgpu

#EXE=$(SCRATCH)/bin/knn.exe
EXE=knn

default: $(EXE)

$(EXE): knn.o timer.o util.o kernel_single.o kernel_bbf.o kernel_opt.o
	nvcc $^ -o $@ $(LFLAG)
	
knn.o: knn.cpp knn.hpp
	$(CC) -c knn.cpp $(CFLAG)

timer.o: timer.cpp timer.hpp
	$(CC) -c timer.cpp $(CFLAG)

util.o: util.cu
	nvcc -c $< $(VFLAG)

kernel_single.o: kernel_single.cu kernel_gpu.hpp
	nvcc -c kernel_single.cu $(VFLAG)

kernel_bbf.o: kernel_bbf.cu kernel_gpu.hpp
	nvcc -c kernel_bbf.cu $(VFLAG)

kernel_opt.o: kernel_opt.cu kernel_gpu.hpp
	nvcc -c kernel_opt.cu $(VFLAG) -I $(MGPU_ROOT) --expt-extended-lambda

run: $(EXE)
	$(EXE)

debug: $(EXE)
	$(EXE) -D -n 2 -d 2 -k 2 -m 2

benchmark: $(EXE)
	$(EXE) -s 8192 -k 16

clean:
	rm -f *.o $(EXE)

