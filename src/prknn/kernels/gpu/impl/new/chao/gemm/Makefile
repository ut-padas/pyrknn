
EXE=driver
LIB=libgemm_new.so

default: $(EXE)

CFLAG=-Wall -O3 -I$(EIGEN_ROOT)
VFLAG=-arch=compute_70 -code=sm_70 -Xcompiler -fPIC -I../util/
LFLAG=-L$(TACC_CUDA_LIB) -L$(TACC_CUDA_LIB)/stubs/ -lcusparse -lcublas -lcuda -lcudart \
			-L./ -lgemm_new -L../util -lutil_new

ifeq ($(findstring frontera,$(shell uname -n)),frontera)
  $(info *** on frontera machine ***)
	VFLAG+=-DFRONTERA
endif

KERNEL=gemm.o
OBJS=driver.o 

$(EXE): $(OBJS) $(LIB)
	g++ $^ -o $@ $(LFLAG)

driver.o: driver.cpp
	g++ -c $< $(CFLAG)

$(LIB): $(KERNEL)
	g++ -shared $^ -o $@

gemm.o: gemm.cu
	nvcc -c $< $(VFLAG) 

run: $(EXE)
	./$(EXE)

clean:
	rm -rf *.o *.so $(EXE)


