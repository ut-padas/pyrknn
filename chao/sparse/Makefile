
EXE=spknn
LIB=libspknngpu.so

DEBUG?=0

CC=g++
CFLAG=-Wall -I $(EIGEN_ROOT)
VFLAG=-gencode arch=compute_70,code=sm_70 -shared -Xcompiler -fPIC
LFLAG=-L$(TACC_CUDA_LIB) -L$(TACC_CUDA_LIB)/stubs/ -lcusparse -lcublas -lcudart \
		-L./ -lspknngpu -L../merge/ -lmergegpu -L../sort/ -lsortgpu -L../util/ -lutil \
		-L../readSVM/ -lreadSVM -L../transpose -ltransposegpu

ifeq ($(DEBUG), 1)
  $(info ----------------------)
  $(info *** debugging mode ***)
  $(info ----------------------)
	CFLAG+=-g
	VFLAG+=-g -G
else
	CFLAG+=-O3
endif

ifeq ($(findstring frontera,$(shell uname -n)),frontera)
  $(info *** on frontera machine ***)
	VFLAG+=-DFRONTERA
endif

KERNEL=kernel_gemm.o kernel_dist.o kernel_leaf.o kernel_tree.o kernel_spknn.o
OBJS=spknn.o print.o copy.o

default: $(EXE)

$(EXE): $(OBJS) $(LIB)
	nvcc $^ -o $@ $(LFLAG)
	
$(LIB): $(KERNEL)
	g++ -shared $^ -o $@
		
spknn.o: spknn.cpp spknn.hpp
	$(CC) -c $< $(CFLAG) -I ../util -I ../readSVM

print.o: print.cpp util.hpp
	$(CC) -c $< $(CFLAG)

copy.o: copy.cu util.hpp
	nvcc -c $< $(VFLAG)

kernel_spknn.o: kernel_spknn.cu
	nvcc -c $< $(VFLAG) -I $(MGPU_ROOT) --expt-extended-lambda -I ../merge -I ../util

kernel_tree.o: kernel_tree.cu
	nvcc -c $< $(VFLAG) -I $(MGPU_ROOT) --expt-extended-lambda -I ../util

kernel_leaf.o: kernel_leaf.cu
	nvcc -c $< $(VFLAG) -I $(MGPU_ROOT) --expt-extended-lambda -I ../util

#kernel_test.o: kernel_test.cu
#	nvcc -c $< $(VFLAG) -I $(MGPU_ROOT) --expt-extended-lambda -I ../merge

kernel_dist.o: kernel_dist.cu
	nvcc -c $< $(VFLAG) -I ../util

kernel_gemm.o: kernel_gemm.cu
	nvcc -c $< $(VFLAG)

run: $(EXE)
	$(EXE) -l 8

debug: $(EXE)
	$(EXE) -days 121 -l 9

benchmark: $(EXE)
	$(EXE) -l 11 -sparse 0.1

clean:
	rm -f *.o $(EXE) $(LIB)

