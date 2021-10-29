#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <cuda.h>
#include <cuda_runtime.h>



__global__ void ComputeSearchGuide(int* R_q, int* C_q, int size, int* SearchInd);
__global__ void ComputeDists_guided(int* R_ref, int* C_ref, float* V_ref, int* R_q, int* C_q, float* V_q, int* leafIds, float* Norms_q, float* Norms_ref, int const k_nn, float* KNN_tmp, int const ppl, int* SearchInd, int const size, int const d, int *QId, int const numqsearch);
