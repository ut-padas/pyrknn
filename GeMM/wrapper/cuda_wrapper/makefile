

#CODE_DIR=/work2/07544/ghafouri/frontera/gits/pyrknn/GeMM
CODE_DIR=/scratch/07544/ghafouri/pyrknn/GeMM/
INC_DIR=-I$(CODE_DIR)/include -I$(CUDA_HOME)/samples/common/inc 
DEV_DIR=$(CODE_DIR)/src/cuda


all: dense

sparse:
	nvcc -shared -c ${INC_DIR} $(DEV_DIR)/SFIKNN_nonpermuted_sqDist.cu -Xcompiler -fPIC
	#nvcc -shared -c ${INC_DIR} $(DEV_DIR)/SFIKNN.cu -Xcompiler -fPIC

dense:
	#nvcc -shared -c ${INC_DIR} $(DEV_DIR)/DFIKNN_streams_chSortVer_chMergeHoriz.cu -Xcompiler -fPIC -lcublas
	#nvcc -shared -c ${INC_DIR} $(DEV_DIR)/DFIKNN_streams_chSortVer.cu -Xcompiler -fPIC -lcublas
	nvcc -shared -c ${INC_DIR} $(DEV_DIR)/DFIKNN_nvgraph_chSortVer.cu -Xcompiler -fPIC -lcublas
	#nvcc -shared -c ${INC_DIR} $(DEV_DIR)/DFIKNN_streams.cu -Xcompiler -fPIC -lcublas


clean:
	rm -f *.o 

