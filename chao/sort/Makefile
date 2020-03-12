
EXE=sort
LIB=libsortgpu.so

default: $(EXE)

CFLAG=-g -arch=compute_70 -code=sm_70 -Xcompiler -fPIC \
			-I $(MGPU_ROOT) --expt-extended-lambda

LFLAG=-L$(TACC_CUDA_LIB) -L$(TACC_CUDA_LIB)/stubs/ -lcuda -lcudart -lcublas

KERNEL=kernel_thrust.o kernel_cub.o kernel_mgpu.o kernel_bitonic.o

OBJS=sort.o 

$(EXE): $(OBJS) $(LIB)
	g++ $^ -o $@ $(LFLAG) -L./ -lsortgpu -L../util -lutil

sort.o: sort.cu sort_gpu.hpp
	nvcc -c $< $(CFLAG) 

$(LIB): $(KERNEL)
	g++ -shared $^ -o $@

kernel_thrust.o: kernel_thrust.cu sort_gpu.hpp
	nvcc -c $< $(CFLAG) 

kernel_cub.o: kernel_cub.cu sort_gpu.hpp
	nvcc -c $< $(CFLAG) -I $(CUB_ROOT)

kernel_mgpu.o: kernel_mgpu.cu sort_gpu.hpp
	nvcc -c $< $(CFLAG) -I $(MGPU_ROOT) --expt-extended-lambda

kernel_bitonic.o: kernel_bitonic.cu sort_gpu.hpp
	nvcc -c $< $(CFLAG) 

run: $(EXE)
	./$(EXE)

benchmark: $(EXE)
	./$(EXE) -B -n 1024 -m 65536

clean:
	rm -rf *.o *.so $(EXE)


