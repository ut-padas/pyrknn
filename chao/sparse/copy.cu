#include "util.hpp"

#include <cassert>
#include <cuda_runtime.h>

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        assert(false);                                                         \
    }                                                                          \
}

void copy_spmat_d2h(int m, int n, int nnz, int *dRowPtr, int *dColIdx, float *dVal, 
    int *hRowPtr, int *hColIdx, float *hVal) {
  CHECK_CUDA( cudaMemcpy(hRowPtr, dRowPtr, (m+1)*sizeof(int), cudaMemcpyDeviceToHost) )
  CHECK_CUDA( cudaMemcpy(hColIdx, dColIdx, nnz*sizeof(int),   cudaMemcpyDeviceToHost) )
  CHECK_CUDA( cudaMemcpy(hVal,    dVal,    nnz*sizeof(float), cudaMemcpyDeviceToHost) )
}

void copy(int m, int *dvec, int *hvec) {
  CHECK_CUDA( cudaMemcpy(hvec, dvec, m*sizeof(int), cudaMemcpyDeviceToHost) )
}

void copy(int m, float *dvec, float *hvec) {
  CHECK_CUDA( cudaMemcpy(hvec, dvec, m*sizeof(float), cudaMemcpyDeviceToHost) )
}

