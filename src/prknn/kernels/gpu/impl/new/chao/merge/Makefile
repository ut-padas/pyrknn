
EXE=driver
LIB=libmerge_new.so

PROD?=1
DEBUG?=0

default: $(EXE)

CFLAG=-arch=compute_70 -code=sm_70 -Xcompiler -fPIC

LFLAG=-L$(TACC_CUDA_LIB) -L$(TACC_CUDA_LIB)/stubs/ -lcuda -lcudart \
			-L../util -lutil_new

ifeq ($(PROD), 1)
	CFLAG+=-DPROD
endif

ifeq ($(DEBUG), 1)
	CFLAG+= -g -G
else
	CFLAG+= -O3
endif

KERNEL=merge.o

OBJS=driver.o timer.o

$(EXE): $(OBJS) $(LIB)
	g++ $^ -o $@ $(LFLAG) -L./ -lmerge_new

driver.o: driver.cpp
	g++ -c $< -Wall -O3 -I $(EIGEN_ROOT)

timer.o: timer.cpp timer.hpp
	g++ -c $< -Wall -O3

$(LIB): $(KERNEL)
	g++ -shared $^ -o $@

merge.o: merge.cu merge.hpp
		nvcc -c $< $(CFLAG) -I../util -I../sort -I $(MGPU_ROOT) --expt-extended-lambda 

run: $(EXE)
	./$(EXE)

benchmark: $(EXE)
	./$(EXE) -n 1024 -m 100000 -k 64

clean:
	rm -rf *.o *.so $(EXE)


