
INC_DIR=-I${PWD} -I$(TACC_CUDA_DIR)/samples/common/inc

all: dense

dense: 
	#nvcc -shared -c ${INC_DIR} dfiknn_vcheck3.cu -Xcompiler -fPIC -lcublas
	#nvcc -shared -c ${INC_DIR} dfiknn.cu -Xcompiler -fPIC -lcublas
	#nvcc -shared -c ${INC_DIR} dfiknn_SHNP2_v2.cu -Xcompiler -fPIC -lcublas
	#nvcc -shared -c ${INC_DIR} dfiknn_SHNP2_v2.cu -Xcompiler -fPIC -lcublas
	nvcc -shared -c ${INC_DIR} dfiknn_SHNP2_v4.cu -Xcompiler -fPIC -lcublas
	#nvcc -shared -c ${INC_DIR} dfiknn_SHNP2_SVNP2.cu -Xcompiler -fPIC -lcublas
	#nvcc -shared -c ${INC_DIR} dfiknn_prof.cu -Xcompiler -fPIC -lcublas
	#mv dfiknn_prof.o dfiknn.o
	#file=profiler
	#${TACC_CUDA_DIR}/nsight-compute-2019.4.0/nv-nsight-cu-cli -f --datails-all --nvtx --export profiler_report dfiknn.o 
	#nvcc -shared -c ${INC_DIR} dfiknn_SortHoriz.cu -Xcompiler -fPIC -lcublas
	#mv dfiknn_SortHoriz.o dfiknn.o
	#mv dfiknn_SHNP2_v2.o dfiknn.o
	mv dfiknn_SHNP2_v4.o dfiknn.o
	#mv dfiknn_SHNP2_SVNP2.o dfiknn.o

clean:
	rm -f *.o
