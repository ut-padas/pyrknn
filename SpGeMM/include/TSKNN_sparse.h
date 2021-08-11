#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <algorithm>

__global__ void TSKNN_compute_norm(int* R, int* C, float* V, int* G_Id, float* Norms, int ppl);
__global__ void compute_dist(int* R, int* C, float* V, int* G_Id, float* Norms, float* K, int m , int k_nn, int ppl, int leaf_batch_g, int max_nnz, int M);
__global__ void find_neighbor(float* knn, int* knn_Id, float* K, int* G_Id, int k, int ppl, int m, int leaf_batch_g, int M);
void TSKNN_gpu(int *R, int *C, float *V, int *G_Id, int M, int leaves, int k, float *knn, int *knn_Id, int max_nnz);


