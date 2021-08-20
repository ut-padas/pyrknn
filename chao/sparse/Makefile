
EXE=driver
LIB=libspknngpu_new.so

DEP=../util/libutil_new.so ../readSVM/libreadSVM_new.so ../transpose/libtranspose_new.so ../gemm/libgemm_new.so \
		../reorder/libreorder_new.so ../orthogonal/liborthogonal_new.so ../merge/libmerge_new.so

DEBUG?=0

CC=g++
CFLAG=-Wall -I $(EIGEN_ROOT) -std=c++14
VFLAG=-gencode arch=compute_70,code=sm_70 -shared -Xcompiler -fPIC -std=c++14
LFLAG=-L$(TACC_CUDA_LIB) -L$(TACC_CUDA_LIB)/stubs/ -lcusparse -lcublas -lcusolver -lcudart \
		-L./ -lspknngpu_new -L../merge -lmerge_new -L../util/ -lutil_new \
		-L../readSVM/ -lreadSVM_new -L../transpose -ltranspose_new -L../gemm -lgemm_new \
		-L../reorder -lreorder_new -L../orthogonal -lorthogonal_new

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

../util/libutil_new.so:
	cd ../util; make -j8

../readSVM/libreadSVM_new.so:
	cd ../readSVM; make -j8

../transpose/libtranspose_new.so:
	cd ../transpose; make -j8

../gemm/libgemm_new.so:
	cd ../gemm; make -j8

../reorder/libreorder_new.so:
	cd ../reorder; make -j8

../orthogonal/liborthogonal_new.so:
	cd ../orthogonal; make -j8

../merge/libmerge_new.so:
	cd ../merge; make -j8

run: $(EXE)
	$(EXE) 

debug: $(EXE)
	$(PROF) $(EXE) -n 10 -d 5 -l 2 -k 3 -bt 1 -t 5 -bl 1

benchmark: $(EXE)
	#$(EXE) -dataset avazu_app_t -l 10 -k 5 -t 3 -bl 512 -bp 256 -bt 1
	#$(EXE) -dataset avazu_app -l 13 -k 16 -t 200 -bl 512 -bp 256 -bt 1
	#time $(EXE) -dataset avazu -l 17 -k 5 -t 200 -bl 1024 -bp 256 -bt 1
	#$(PROF) $(EXE) -dataset avazu -l 15 -k 5 -t 2 -bl 1024 -bp 64 -bt 1
	#$(EXE) -n 524288 -d 10000 -sparse 0.0001 -k 64 -t 1 -bl 10 -bp 64 -bt 1 -l 8
	#$(EXE) -dataset test -k 64 -t 1 -bl 512 -bp 64 -bt 1 -l 8
	#
	#$(EXE) -dataset url -l 10 -k 64 -t 300 -bl 512 -bp 128 -bt 1
	$(EXE) -dataset avazu -l 15 -k 5 -t 300 -bl 1024 -bp 128 -bt 1

clean:
	rm -f *.o $(EXE) $(LIB)

