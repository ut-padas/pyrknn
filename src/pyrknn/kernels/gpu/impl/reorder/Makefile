
EXE=driver
LIB=libreorder.so

default: $(EXE)

CFLAG=-O3 -arch=compute_70 -code=sm_70 -Xcompiler -fPIC
LFLAG=-L$(TACC_CUDA_LIB) -L$(TACC_CUDA_LIB)/stubs/ -lcusparse -lcublas -lcuda -lcudart \
			-L./ -lreorder -L../gemm/ -lgemm -L../util -lutil

ifeq ($(findstring frontera,$(shell uname -n)),frontera)
  $(info *** on frontera machine ***)
	CFLAG+=-DFRONTERA
endif

KERNEL=reorder.o
OBJS=driver.o 

$(EXE): $(OBJS) $(LIB)
	g++ $^ -o $@ $(LFLAG)

driver.o: driver.cpp
	g++ -c $< -Wall -O3 -I $(EIGEN_ROOT)

$(LIB): $(KERNEL)
	g++ -shared $^ -o $@

reorder.o: reorder.cu reorder.hpp
	nvcc -c $< $(CFLAG) -I../util -I../gemm 

run: $(EXE)
	./$(EXE)

clean:
	rm -rf *.o *.so $(EXE)


