
EXE=driver
LIB=libsort_new.so

default: $(EXE)

CFLAG=-O3 -arch=compute_70 -code=sm_70 -Xcompiler -fPIC
LFLAG=-L$(TACC_CUDA_LIB) -L$(TACC_CUDA_LIB)/stubs/ -lcuda -lcudart

KERNEL=sort.o
OBJS=driver.o sort.o

$(EXE): $(OBJS) $(LIB)
	g++ $^ -o $@ $(LFLAG)

driver.o: driver.cpp
	g++ -c $< -Wall -O3 -I $(EIGEN_ROOT)

$(LIB): $(KERNEL)
	g++ -shared $^ -o $@

sort.o: sort.cu sort.hpp
	nvcc -c $< $(CFLAG) -I../util -I $(MGPU_ROOT) --expt-extended-lambda

run: $(EXE)
	./$(EXE)

clean:
	rm -rf *.o *.so $(EXE)


