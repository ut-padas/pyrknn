
EXE=knn
LIB=libknngpu.so

PROD=1

CC=g++
CFLAG=-O3 -Wall -I $(EIGEN_ROOT)
VFLAG=-O3 -gencode arch=compute_70,code=sm_70 -shared -Xcompiler -fPIC
LFLAG=-L$(TACC_CUDA_LIB) -L$(TACC_CUDA_LIB)/stubs/ -lcuda -lcudart -lcublas \
			-L./sort/ -lsortgpu -L./merge/ -lmergegpu -L./util/ -lutil -L./ -lknngpu

ifeq ($(PROD), 1)
	VFLAG+=-DPROD
endif


KERNEL=kernel_knn.o

default: $(EXE)

$(EXE): knn.o $(LIB)
	nvcc $^ -o $@ $(LFLAG)
	
$(LIB): $(KERNEL)
	g++ -shared $^ -o $@
		
knn.o: knn.cpp knn_gpu.hpp
	$(CC) -c $< $(CFLAG)

kernel_knn.o: kernel_knn.cu knn_gpu.hpp
	nvcc -c $< $(VFLAG) -I $(MGPU_ROOT) --expt-extended-lambda

run: $(EXE)
	$(EXE)

debug: $(EXE)
	$(EXE) -D -n 10 -d 2 -k 5 -m 2 -s 2

benchmark: $(EXE)
	$(EXE) -s 8192 -k 16

clean:
	rm -f *.o $(EXE) $(LIB)

