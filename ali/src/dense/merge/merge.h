
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void S_PrecompMergeNP2(int* SortInd, int* StepLen, int* StepStart, int* tidIdMap, int* tidSortDir, int steps);
__global__ void S_MergeHorizNP2(float* KNN, int* KNN_Id, int k_nn, int ppl, float* tmp_knnDist, int* glob_leafIds, int steps, int* QId, int* SortInd, int* step_len, int* step_start, int* tid_idmap, int* tid_sortdir, int* local_leafIds, float* Norms_q, float* Norms_ref);


