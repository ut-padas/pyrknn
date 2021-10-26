#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>


__global__ void ComputeNorms(int * R, int* C, float* V, float* Norms);
__global__ void ComputeNorms_ref(int * R, int* C, float* V, float* Norms, int* G_leafIds);
