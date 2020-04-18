
EXE=driver
LIB=libdenknngpu_new.so

DEP=../util/libutil_new.so ../readSVM/libreadSVM_new.so ../transpose/libtranspose_new.so \
		../reorder/libreorder_new.so ../orthogonal/liborthogonal_new.so ../merge/libmerge_new.so \
		../gemm/libgemm_new.so

DEBUG?=0

CC=g++
CFLAG=-Wall -I $(EIGEN_ROOT)
VFLAG=-gencode arch=compute_70,code=sm_70 -shared -Xcompiler -fPIC
LFLAG=-L$(TACC_CUDA_LIB) -L$(TACC_CUDA_LIB)/stubs/ -lcusparse -lcublas -lcusolver -lcudart \
		-L./ -ldenknngpu_new -L../merge -lmerge_new -L../util/ -lutil_new -L../gemm -lgemm_new \
		-L../readSVM/ -lreadSVM_new -L../reorder -lreorder_new \
		-L../orthogonal -lorthogonal_new

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

KERNEL=denknn.o build_tree.o leaf_knn.o
OBJS=driver.o

default: $(EXE)

$(EXE): $(OBJS) $(LIB) $(DEP) 
	nvcc $(OBJS) $(LIB) -o $@ $(LFLAG)
	
$(LIB): $(KERNEL)
	g++ -shared $^ -o $@
		
driver.o: driver.cpp
	$(CC) -c $< $(CFLAG) -I ../util -I ../readSVM

denknn.o: denknn.cu
	nvcc -c $< $(VFLAG) -I ../merge -I ../util -I ../reorder -I ../orthogonal -I ../gemm

build_tree.o: build_tree.cu
	nvcc -c $< $(VFLAG) -I $(MGPU_ROOT) --expt-extended-lambda -I ../util -I ../reorder

leaf_knn.o: leaf_knn.cu
	nvcc -c $< $(VFLAG) -I ../util -I ../sort -I $(MGPU_ROOT) --expt-extended-lambda

../util/libutil_new.so:
	cd ../util; make -j8

../readSVM/libreadSVM_new.so:
	cd ../readSVM; make -j8

../transpose/libtranspose_new.so:
	cd ../transpose; make -j8

../orthogonal/liborthogonal_new.so:
	cd ../orthogonal; make -j8

../gemm/libgemm_new.so:
	cd ../gemm; make -j8

../reorder/libreorder_new.so:
	cd ../reorder; make -j8

../merge/libmerge_new.so:
	cd ../merge; make -j8

run: $(EXE)
	$(EXE) 

debug: $(EXE)
	$(EXE) -n 10 -d 5 -l 1 -k 3 -bt 1 -t 20

benchmark: $(EXE)
	#$(EXE) -dataset mnist -l 5 -k 64 -bt 1 -t 50
	#$(EXE) -k 64 -bt 1 -t 2 -l 10 -leaf 1024 -bp 256 -bl 1024
	$(EXE) -k 64 -bt 1 -t 2 -l 12 -leaf 4096 -bp 64 -d 10
	
clean:
	rm -f *.o $(EXE) $(LIB)

