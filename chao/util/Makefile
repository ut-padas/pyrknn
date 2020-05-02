
LIB=libutil_new.so

default: $(LIB)

CC=g++ -std=c++11
CFLAG=-O3 -arch=compute_70 -code=sm_70 -Xcompiler -fPIC -std=c++11

OBJS=rand.o timer.o timer_gpu.o print.o dprint.o copy.o 

rand.o: rand.cu rand.hpp
	nvcc -c $< $(CFLAG) 

timer_gpu.o: timer_gpu.cu timer_gpu.hpp
	nvcc -c $< $(CFLAG)

copy.o: copy.cu
	nvcc -c $< $(CFLAG)

dprint.o: dprint.cu
	nvcc -c $< $(CFLAG)

print.o: print.cpp
	$(CC) -c $< -Wall -O3 -fPIC -I$(EIGEN_ROOT)

timer.o: timer.cpp timer.hpp
	$(CC) -c $< -Wall -O3 -fPIC

$(LIB): $(OBJS)
	g++ -shared $^ -o $@ 

clean:
	rm -rf *.o *.so


