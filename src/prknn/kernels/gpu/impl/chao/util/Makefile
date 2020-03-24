
LIB=libutil.so

default: $(LIB)

CFLAG=-O3 -arch=compute_70 -code=sm_70 -Xcompiler -fPIC

OBJS=rand.o timer.o timer_gpu.o

rand.o: rand.cu rand.hpp
	nvcc -c $< $(CFLAG) 

timer_gpu.o: timer_gpu.cu timer_gpu.hpp
	nvcc -c $< $(CFLAG)

timer.o: timer.cpp timer.hpp
	g++ -c $< -Wall -O3

$(LIB): $(OBJS)
	g++ -shared $^ -o $@

clean:
	rm -rf *.o *.so


