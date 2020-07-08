
EXE=driver
LIB=libreorder_new.so

default: $(EXE)

CFLAG=-O3 -arch=compute_70 -code=sm_70 -Xcompiler -fPIC -std=c++11
LFLAG=-L$(TACC_CUDA_LIB) -L$(TACC_CUDA_LIB)/stubs/ -lcusparse -lcublas -lcuda -lcudart \
			-L./ -lreorder_new -L../gemm/ -lgemm_new -L../util -lutil_new

ifeq ($(findstring frontera,$(shell uname -n)),frontera)
  $(info *** on frontera machine ***)
	CFLAG+=-DFRONTERA
endif

KERNEL=reorder.o
OBJS=driver.o 

$(EXE): $(OBJS) $(LIB)
	g++ $^ -o $@ $(LFLAG)

driver.o: driver.cpp
	g++ -std=c++11 -c $< -Wall -O3 -I $(EIGEN_ROOT)

$(LIB): $(KERNEL)
	g++ -shared $^ -o $@

reorder.o: reorder.cu reorder.hpp
	nvcc -c $< $(CFLAG) -I../util -I../gemm 

run: $(EXE)
	./$(EXE)

clean:
	rm -rf *.o *.so $(EXE)


