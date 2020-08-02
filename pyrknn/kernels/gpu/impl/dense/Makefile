
EXE=driver
LIB=libdenknngpu.so

DEP=../util/libutil.so ../readSVM/libreadSVM.so ../transpose/libtranspose.so \
		../reorder/libreorder.so ../orthogonal/liborthogonal.so ../merge/libmerge.so \
		../gemm/libgemm.so

DEBUG=0

CC=g++
CFLAG=-Wall -I $(EIGEN_ROOT)
VFLAG= -Xptxas -v, -O3 -gencode arch=compute_70,code=sm_70 --verbose -shared -Xcompiler -fPIC
LFLAG=-L$(TACC_CUDA_LIB) -L$(TACC_CUDA_LIB)/stubs/ -lcusparse -lcublas -lcusolver -lcudart \
		-L./ -ldenknngpu -L../merge -lmerge -L../util/ -lutil -L../gemm -lgemm \
		-L../readSVM/ -lreadSVM -L../reorder -lreorder \
		-L../orthogonal -lorthogonal

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

../util/libutil.so:
	make -C ../util -j8

../readSVM/libreadSVM.so: ../util/libutil.so
	make -C ../readSVM -j8

../transpose/libtranspose.so: ../util/libutil.so
	make -C ../transpose -j8

../orthogonal/liborthogonal.so: ../util/libutil.so
	make -C ../orthogonal -j8

../gemm/libgemm.so: ../util/libutil.so
	make -C ../gemm -j8

../reorder/libreorder.so: ../gemm/libgemm.so
	make -C ../reorder -j8

../merge/libmerge.so: ../util/libutil.so ../sort/libsort.so
	make -C ../merge -j8

../sort/libsort.so: ../util/libutil.so
	make -C ../sort -j8

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

