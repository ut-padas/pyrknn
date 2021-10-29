#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>



__global__ void FusedLeafKNN_seq(int* R_ref, int* C_ref, float* V_ref, int* R_q, int* C_q, float* V_q, int* local_leafIds, float* Norms_q, float* Norms_ref, int const k_nn, int const ppl, int const d, int *QId, float* NDist, int* NId, int* glob_leafIds, int const steps, int* SortInd, int* StepLen, int* StepStart, int* tidIdMap, int* tidSortDir);
