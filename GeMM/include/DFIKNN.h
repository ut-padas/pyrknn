#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <algorithm>
#include <cublas_v2.h>

//__global__ void FIKNN_compute_norm_dense(float* data, int* G_Id, float* Norms, int ppl, int d);
/*
__global__ void FIKNN_tri_dense(float* data, int* G_Id, float* Norms , int k_nn, float* KNN_dist, int* KNN_Id, int ppl, int d);

__global__ void FIKNN_kernel_A_dense(float *data, int* G_Id, float* Norms, int k_nn, float* KNN_dist, int* KNN_Id, int ppl, int d, int blockInd, int* sort_arr, int* sort_arr_part, int steps, float* d_knn_temp);

__global__ void FIKNN_kernel_B_dense(float* KNN, int* KNN_Id, int k_nn, int ppl, int blockInd, float* d_temp_knn,int* G_Id);
*/

static const char *cudaGetErrorEnum(cublasStatus_t error) {
    switch (error) {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";

        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";

        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";

        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";

        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";

        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";

        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";

        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";
    }
    return "<unknown>";
}


#define CHECK_CUBLAS(ans) { cublasAssert((ans), __FILE__, __LINE__); }
inline void cublasAssert(cublasStatus_t code, const char *file, int line, bool abort=true) {
   if (code != CUBLAS_STATUS_SUCCESS) {
      fprintf(stderr,"CUBLAS assert: %s %s %d\n", cudaGetErrorEnum(code), file, line);
      if (abort) exit(code);
   }
}



void precomp_arbsize_sortId_dense(int* d_arr, int* d_arr_part, int N_true, int N_pow2, int steps, int copy_size);

void dfi_leafknn(float *data, int *G_Id, int M, int leaves, int k, float *knn, int *knn_Id, int dim);
