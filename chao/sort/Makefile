
EXE=sort

default: $(EXE)

CFLAG=-use_fast_math -O3 -arch=compute_70 -code=sm_70 -shared -Xcompiler -fPIC
LFLAG=-L$(TACC_CUDA_LIB) -L$(TACC_CUDA_LIB)/stubs/ -lcuda -lcudart -lcublas

$(EXE): sort.o timer.o kernel_thrust.o kernel_cub.o kernel_mgpu.o kernel_bitonic.o
	g++ $^ -o $@ $(LFLAG)

sort.o: sort.cu
	nvcc -c $^ $(CFLAG) 

kernel_thrust.o: kernel_thrust.cu
	nvcc -c $^ $(CFLAG) 

kernel_cub.o: kernel_cub.cu
	nvcc -c $^ $(CFLAG) -I $(CUB_ROOT)

kernel_mgpu.o: kernel_mgpu.cu
	nvcc -c $^ $(CFLAG) -I $(MGPU_ROOT) --expt-extended-lambda

kernel_bitonic.o: kernel_bitonic.cu
	nvcc -c $^ $(CFLAG) 

timer.o: timer.cpp
	g++ $^ -c -O3

run: $(EXE)
	./$(EXE)

benchmark: $(EXE)
	./$(EXE) -B -n 1024 -m 65536

clean:
	rm -rf *.o $(EXE)


