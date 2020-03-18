
EXE=merge
LIB=libmergegpu.so
PROD=1

default: $(EXE)

CFLAG=-O3 -arch=compute_70 -code=sm_70 -Xcompiler -fPIC

LFLAG=-L$(TACC_CUDA_LIB) -L$(TACC_CUDA_LIB)/stubs/ -lcuda -lcudart

ifeq ($(PROD), 1)
	CFLAG+=-DPROD
endif

KERNEL=kernel_merge.o

OBJS=merge.o timer.o

$(EXE): $(OBJS) $(LIB)
	g++ $^ -o $@ $(LFLAG) -L./ -lmergegpu -L../sort -lsortgpu -L../util -lutil

merge.o: merge.cpp
	g++ -c $< -Wall -O3 -I $(EIGEN_ROOT)

timer.o: timer.cpp timer.hpp
	g++ -c $< -Wall -O3

$(LIB): $(KERNEL)
	g++ -shared $^ -o $@

kernel_merge.o: kernel_merge.cu merge_gpu.hpp
		nvcc -c $< $(CFLAG) -I $(MGPU_ROOT) --expt-extended-lambda 

run: $(EXE)
	./$(EXE)

benchmark: $(EXE)
	./$(EXE) -n 1024 -m 100000 -k 64

clean:
	rm -rf *.o *.so $(EXE)


