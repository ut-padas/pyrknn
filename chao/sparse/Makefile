
EXE=spknn
LIB=libspknngpu.so

DEBUG?=1

CC=g++
CFLAG=-Wall -I $(EIGEN_ROOT)
VFLAG=-gencode arch=compute_70,code=sm_70 -shared -Xcompiler -fPIC
LFLAG=-L$(TACC_CUDA_LIB) -L$(TACC_CUDA_LIB)/stubs/ -lcusparse -lcublas -lcudart \
			-L./ -lspknngpu -L../merge/ -lmergegpu -L../sort/ -lsortgpu -L../util/ -lutil

ifeq ($(DEBUG), 1)
	CFLAG+=-g
	VFLAG+=-g -G
else
	CFLAG+=-O3
	vFLAG+=-O3
endif

KERNEL=kernel_gemm.o kernel_leaf.o kernel_spknn.o kernel_test.o
OBJS=spknn.o print.o copy.o

default: $(EXE)

$(EXE): $(OBJS) $(LIB)
	nvcc $^ -o $@ $(LFLAG)
	
$(LIB): $(KERNEL)
	g++ -shared $^ -o $@
		
spknn.o: spknn.cpp spknn.hpp
	$(CC) -c $< $(CFLAG)

print.o: print.cpp util.hpp
	$(CC) -c $< $(CFLAG)

copy.o: copy.cu util.hpp
	nvcc -c $< $(VFLAG)

kernel_spknn.o: kernel_spknn.cu
	nvcc -c $< $(VFLAG) -I $(MGPU_ROOT) --expt-extended-lambda -I ../merge -I ../util

kernel_leaf.o: kernel_leaf.cu
	nvcc -c $< $(VFLAG) -I $(MGPU_ROOT) --expt-extended-lambda -I ../merge

kernel_test.o: kernel_test.cu
	nvcc -c $< $(VFLAG) -I $(MGPU_ROOT) --expt-extended-lambda

kernel_gemm.o: kernel_gemm.cu
	nvcc -c $< $(VFLAG) -I $(MGPU_ROOT) --expt-extended-lambda

run: $(EXE)
	$(EXE) -s 1

debug: $(EXE)
	$(EXE) -n 8 -d 3 -m 2 -l 2 -k 1 -sparse 0.7

clean:
	rm -f *.o $(EXE) $(LIB)

