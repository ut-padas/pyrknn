
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
	cd ../util; make -j8

../readSVM/libreadSVM.so:
	cd ../readSVM; make -j8

../transpose/libtranspose.so:
	cd ../transpose; make -j8

../gemm/libgemm.so:
	cd ../gemm; make -j8

../reorder/libreorder.so:
	cd ../reorder; make -j8

../orthogonal/liborthogonal.so:
	cd ../orthogonal; make -j8

../merge/libmerge.so:
	cd ../merge; make -j8

run: $(EXE)
	$(EXE) 

debug: $(EXE)
	$(EXE) -n 10 -d 5 -l 2 -k 3 -bt 1 -t 10 -bl 1

benchmark: $(EXE)
	$(EXE) -dataset avazu -l 13 -k 5 -t 20 -bl 512 -bp 64 -bt 1
	#$(EXE) -dataset url -l 9 -k 10 -t 50 -bl 512 -bp 64 -bt 1

clean:
	rm -f *.o $(EXE) $(LIB)

