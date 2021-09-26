

#CODE_DIR=/work2/07544/ghafouri/frontera/gits/pyrknn/GeMM
INC_DIR=-I$(CUDA_HOME)/samples/common/inc 


all: sort

sort:
	nvcc -shared -c ${INC_DIR} SortMerge_v2.cu -Xcompiler -fPIC


clean:
	rm -f *.o 

