#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>


__global__ void ComputeNorms(int* R_q, int* C_q, float* V_q, float* Norms_q);
__global__ void ComputeNormRef(int* R_ref, int* C_ref, float* V_ref, float* Norms_ref, int* local_leafIds);
