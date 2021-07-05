
EXE=driver
LIB=libtranspose.so

default: $(EXE)

CFLAG=-O3 -arch=compute_70 -code=sm_70 -Xcompiler -fPIC
LFLAG=-L$(TACC_CUDA_LIB) -L$(TACC_CUDA_LIB)/stubs/ -lcuda -lcudart \
			-L./ -ltranspose -L../util -lutil


KERNEL=transpose.o
OBJS=driver.o 

$(EXE): $(OBJS) $(LIB)
	g++ $^ -o $@ $(LFLAG)

driver.o: driver.cpp
	g++ -c $< -Wall -O3 -I $(EIGEN_ROOT)

$(LIB): $(KERNEL)
	g++ -shared $^ -o $@

transpose.o: transpose.cu
	nvcc -c $< $(CFLAG) -I../util 

run: $(EXE)
	./$(EXE)

clean:
	rm -rf *.o *.so $(EXE)


