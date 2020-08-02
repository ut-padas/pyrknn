
EXE=driver
LIB=libspknngpu.so

DEP=../util/libutil.so ../readSVM/libreadSVM.so ../transpose/libtranspose.so ../gemm/libgemm.so \
		../reorder/libreorder.so ../orthogonal/liborthogonal.so ../merge/libmerge.so

DEBUG?=0

CC=g++
CFLAG=-Wall -I $(EIGEN_ROOT)
VFLAG=-gencode arch=compute_70,code=sm_70 -shared -Xcompiler -fPIC
LFLAG=-L$(TACC_CUDA_LIB) -L$(TACC_CUDA_LIB)/stubs/ -lcusparse -lcublas -lcusolver -lcudart \
		-L./ -lspknngpu -L../merge -lmerge -L../util/ -lutil \
		-L../readSVM/ -lreadSVM -L../transpose -ltranspose -L../gemm -lgemm \
		-L../reorder -lreorder -L../orthogonal -lorthogonal

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

KERNEL=spknn.o build_tree.o leaf_knn.o find_nbor.o
OBJS=driver.o

PROF=nvprof --metrics flop_count_sp --metrics dram_read_throughput --metrics dram_write_throughput \
		 --profile-from-start off

default: $(EXE)

$(EXE): $(OBJS) $(LIB) $(DEP) 
	nvcc $(OBJS) $(LIB) -o $@ $(LFLAG)
	
$(LIB): $(KERNEL)
	g++ -shared $^ -o $@
		
driver.o: driver.cpp
	$(CC) -c $< $(CFLAG) -I ../util -I ../readSVM

spknn.o: spknn.cu
	nvcc -c $< $(VFLAG) -I ../merge -I ../util -I ../reorder -I ../gemm -I ../orthogonal

build_tree.o: build_tree.cu
	nvcc -c $< $(VFLAG) -I $(MGPU_ROOT) --expt-extended-lambda -I ../util -I ../reorder

leaf_knn.o: leaf_knn.cu
	nvcc -c $< $(VFLAG) -I ../util -I ../gemm -I ../transpose

find_nbor.o: find_nbor.cu
	nvcc -c $< $(VFLAG) -I $(MGPU_ROOT) --expt-extended-lambda -I../util

../util/libutil.so:
	make -C ../util -j8

../readSVM/libreadSVM.so: ../util/libutil.so
	make -C ../readSVM -j8

../transpose/libtranspose.so: ../util/libutil.so 
	make -C ../transpose -j8

../gemm/libgemm.so: ../util/libutil.so
	make -C ../gemm -j8

../reorder/libreorder.so: ../gemm/libgemm.so
	make -C ../reorder -j8

../orthogonal/liborthogonal.so: ../util/libutil.so
	make -C ../orthogonal -j8

../merge/libmerge.so: ../sort/libsort.so ../util/libutil.so
	make -C ../merge -j8 

../sort/libsort.so: ../util/libutil.so
	make -C ../sort -j8

run: $(EXE)
	$(EXE) 

debug: $(EXE)
	$(PROF) $(EXE) -n 10 -d 5 -l 2 -k 3 -bt 1 -t 5 -bl 1

benchmark: $(EXE)
	#$(EXE) -dataset avazu_app_t -l 10 -k 5 -t 3 -bl 512 -bp 256 -bt 1
	#$(EXE) -dataset avazu_app -l 13 -k 16 -t 200 -bl 512 -bp 256 -bt 1
	#time $(EXE) -dataset avazu -l 17 -k 5 -t 200 -bl 1024 -bp 256 -bt 1
	$(EXE) -dataset url -l 11 -k 10 -t 3 -bl 512 -bp 64 -bt 1
	#$(PROF) $(EXE) -dataset avazu -l 15 -k 5 -t 2 -bl 1024 -bp 64 -bt 1

clean:
	rm -f *.o $(EXE) $(LIB)

