
EXE=sort

default: $(EXE)

CFLAG=-O3 -arch=compute_70 -code=sm_70 -shared -Xcompiler -fPIC
LFLAG=-L$(TACC_CUDA_LIB) -L$(TACC_CUDA_LIB)/stubs/ -lcuda -lcudart -lcublas

$(EXE): sort.o timer.o
	g++ $^ -o $@ $(LFLAG)

sort.o: sort.cu
	nvcc -c $^ $(CFLAG) 

timer.o: timer.cpp
	g++ $^ -o $@ $(CFLAG)

run: $(EXE)
	./$(EXE)

clean:
	rm -rf *.o $(EXE)


