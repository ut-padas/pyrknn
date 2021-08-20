
EXE=driver
LIB=liborthogonal_new.so

default: $(EXE)

CFLAG=-O3 -arch=compute_70 -code=sm_70 -Xcompiler -fPIC -std=c++14
LFLAG=-L$(TACC_CUDA_LIB) -L$(TACC_CUDA_LIB)/stubs/ -lcuda -lcudart -lcusolver \
			-L./ -lorthogonal_new

KERNEL=orthogonal.o
OBJS=driver.o 

$(EXE): $(OBJS) $(LIB)
	g++ $^ -o $@ $(LFLAG)

driver.o: driver.cpp
	g++ -std=c++14 -c $< -Wall -O3 -I $(EIGEN_ROOT)

$(LIB): $(KERNEL)
	g++ -shared $^ -o $@

orthogonal.o: orthogonal.cu
	nvcc -c $< $(CFLAG) -I../util 

run: $(EXE)
	./$(EXE)

clean:
	rm -rf *.o *.so $(EXE)


