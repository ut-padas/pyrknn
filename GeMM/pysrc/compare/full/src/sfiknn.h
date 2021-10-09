#include <stdio.h>
#include <stdlib.h>
//#include <cuda.h>
//#include <cuda_runtime.h>
//#include <helper_cuda.h>
#include <algorithm>


void gpu_knn(int *R, int *C, float *V, int *G_Id, int M, int leaves, int k, float *KNN, int *KNN_Id);

//void sfi_leafknn(int *R, int *C, float *V, int *G_Id, int M, int leaves, int k, float *knn, int *knn_Id, int deviceId);
