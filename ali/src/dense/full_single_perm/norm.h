#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>


__global__ void ComputeNorms(float* V, float* Norms, int dim);
__global__ void ComputeNorms_ref(float* V, float* Norms, int* G_leafIds, int dim);
