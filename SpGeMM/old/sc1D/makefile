file = SpGeMM_1D_SingleKernel
#file = SpGeMM_1D_TriPart
all:
	nvcc -I${CUDA_HOME}/samples/common/inc -lineinfo $(file).cu -o $(file)
run:
	./$(file)
