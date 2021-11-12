
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void S_PrecompMergeNP2(int* SortInd, int* StepLen, int* StepStart, int* tidIdMap, int* tidSortDir, int steps);


