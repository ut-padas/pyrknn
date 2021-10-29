#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>


__global__ void ComputeNorms(float* X, float* Norms, int dim);
__global__ void ComputeNorms_ref(float* X, float* Norms, int* local_leafIds, int dim);
