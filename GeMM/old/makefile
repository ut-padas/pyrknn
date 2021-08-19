#file = tmp
#file1 = SpGeMM_Iter
file1 = SpGeMM_Iter_stripes
file2 = SpGeMM_TriPart_v1

all:
	nvcc -I${CUDA_HOME}/samples/common/inc -lineinfo $(file1).cu -o $(file1)
	./$(file1)
compile1:
	nvcc -I${CUDA_HOME}/samples/common/inc -lineinfo $(file1).cu -o $(file1)
run1:
	./$(file1)
tri: 
	nvcc -I${CUDA_HOME}/samples/common/inc -lineinfo $(file2).cu -o $(file2)
	./$(file2)
run2:
	./$(file2)

compile2:
	nvcc -I${CUDA_HOME}/samples/common/inc -lineinfo $(file2).cu -o $(file2)

