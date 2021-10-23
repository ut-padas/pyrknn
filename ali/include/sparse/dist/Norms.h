
#include <cuda.h>
#include <cuda_runtime.h>


__global__ void ComputeNorms(int * R, int* C, float* V, float* Norms);
