#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <algorithm>


//__global__ void FIKNN_compute_norm_dense(float* data, int* G_Id, float* Norms, int ppl, int d);

//__global__ void FIKNN_tri_dense(float* data, int* G_Id, float* Norms , int k_nn, float* KNN_dist, int* KNN_Id, int ppl, int d);

//__global__ void FIKNN_kernel_A_dense(float *data, int* G_Id, float* Norms, int k_nn, float* KNN_dist, int* KNN_Id, int ppl, int d, int blockInd, int* sort_arr, int* sort_arr_part, int steps, float* d_knn_temp);

//__global__ void FIKNN_kernel_B_dense(float* KNN, int* KNN_Id, int k_nn, int ppl, int blockInd, float* d_temp_knn,int* G_Id);

//void precomp_arbsize_sortId_dense(int* d_arr, int* d_arr_part, int N_true, int N_pow2, int steps, int copy_size);

void dfi_leafknn(float *data, int *G_Id, int M, int leaves, int k, float *knn, int *knn_Id, int dim);
