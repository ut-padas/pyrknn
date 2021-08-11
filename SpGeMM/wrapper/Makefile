


all: test
	

simple: dev/simple_cuda.o
	./build.sh

dev/simple_cuda.o: clean
	nvcc -shared -c -I$(CUDA_HOME)/samples/common/inc dev/simple.cu -o dev/simple_cuda.o -Xcompiler -fPIC

test: simple
	./test.sh

clean:
	rm -f dev/*.o
	rm -rf cuda_wrapper
