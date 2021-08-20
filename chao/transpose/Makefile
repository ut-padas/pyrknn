
EXE=driver
LIB=libtranspose_new.so

default: $(EXE)

CFLAG=-O3 -arch=compute_70 -code=sm_70 -Xcompiler -fPIC -std=c++14
LFLAG=-L$(TACC_CUDA_LIB) -L$(TACC_CUDA_LIB)/stubs/ -lcuda -lcudart \
			-L./ -ltranspose_new -L../util -lutil_new

KERNEL=transpose.o
OBJS=driver.o 

$(EXE): $(OBJS) $(LIB)
	g++ $^ -o $@ $(LFLAG)

driver.o: driver.cpp
	g++ -std=c++11 -c $< -Wall -O3 -I $(EIGEN_ROOT)

$(LIB): $(KERNEL)
	g++ -shared $^ -o $@

transpose.o: transpose.cu
	nvcc -c $< $(CFLAG) -I../util 

run: $(EXE)
	./$(EXE)

clean:
	rm -rf *.o *.so $(EXE)


