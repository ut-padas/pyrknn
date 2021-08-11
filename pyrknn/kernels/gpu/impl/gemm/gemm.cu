#include "util_gpu.hpp"
#include "knn_handle.hpp"
#include "timer_gpu.hpp"
#include "cuda_profiler_api.h"

// (dense) A * (dense) B = (dense) C
void GEMM(int m, int n, int k, const fvec &A, const fvec &B, fvec &C) {
  auto const& handle = knnHandle_t::instance();
  float alpha = 1.0;
  float beta = 0.;
  CHECK_CUBLAS( cublasSgemm(handle.blas, CUBLAS_OP_T, CUBLAS_OP_N, 
        m, n, k, &alpha,
        thrust::raw_pointer_cast(A.data()), k,
        thrust::raw_pointer_cast(B.data()), k, &beta,
        thrust::raw_pointer_cast(C.data()), m) );
}


void gemm_gpu(int m, int n, int k, const float *hA, const float *hB, float *hC) {
  fvec dA(hA, hA+m*k);
  fvec dB(hB, hB+k*n);
  fvec dC(m*n);
  GEMM(m, n, k, dA, dB, dC);
  thrust::copy_n(dC.begin(), m*n, hC);
}


// (sparse) A * (dense) B = (dense) C
void GEMM_SDD(int m, int n, int k,
    dvec<int> &rowPtrA, dvec<int> &colIdxA, dvec<float> &valA, int nnzA,
    dvec<float> &B, dvec<float> &C) {
  
  const float alpha = 1.0;
  const float beta = 0.0;
  auto const& handle = knnHandle_t::instance();
  
  int *rowPtr = thrust::raw_pointer_cast(rowPtrA.data());
  int *colIdx = thrust::raw_pointer_cast(colIdxA.data());
  float *val  = thrust::raw_pointer_cast(valA.data());
  float *valB = thrust::raw_pointer_cast(B.data());
  float *valC = thrust::raw_pointer_cast(C.data());

  CHECK_CUSPARSE( cusparseScsrmm(
        handle.sparse,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        m, n, k, nnzA,
        &alpha, handle.mat,
        val, rowPtr, colIdx, 
        valB, k,
        &beta, valC, m) )
}

/*
   // For some reason LDB seems to be 1 in SpMM routine
void GEMM_SDD(int m, int n, int k,
    dvec<int> &rowPtrA, dvec<int> &colIdxA, dvec<float> &valA, int nnzA,
    dvec<float> &B, dvec<float> &C) {

  const float alpha = 1.0;
  const float beta = 0.0;
  auto const& handle = knnHandle_t::instance();
                       
  int *rowPtr = thrust::raw_pointer_cast(rowPtrA.data());
  int *colIdx = thrust::raw_pointer_cast(colIdxA.data());
  float *val = thrust::raw_pointer_cast(valA.data());
  
  cusparseSpMatDescr_t matA;
  CHECK_CUSPARSE( cusparseCreateCsr(&matA, m, k, nnzA,
                                    rowPtr, colIdx, val,
                                    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                    CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) )

  float *valB = thrust::raw_pointer_cast(B.data());
  float *valC = thrust::raw_pointer_cast(C.data());
 

  std::cout<<"k: "<<k<<", n: "<<n<<std::endl;
  tprint(n,k,B,"dB");

  dprint(m, k, nnzA, rowPtr, colIdx, val, "dA");

  cusparseDnMatDescr_t matB, matC;
  CHECK_CUSPARSE( cusparseCreateDnMat(&matB, k, n, k, valB, CUDA_R_32F, CUSPARSE_ORDER_COL) )
  CHECK_CUSPARSE( cusparseCreateDnMat(&matC, m, n, m, valC, CUDA_R_32F, CUSPARSE_ORDER_COL) )

int64_t rowB, colB, ldB;
void *vB;
cudaDataType dt;
cusparseOrder_t od;
CHECK_CUSPARSE( cusparseDnMatGet(matB, &rowB, &colB, &ldB, &vB, &dt, &od) )

std::cout<<"row: "<<rowB<<", col: "<<colB<<", ldB: "<<ldB<<std::endl;

  size_t bufferSize;
  CHECK_CUSPARSE( cusparseSpMM_bufferSize(
                        handle.sparse,
                        CUSPARSE_OPERATION_NON_TRANSPOSE,
                        CUSPARSE_OPERATION_NON_TRANSPOSE,
                        &alpha,
                        matA,
                        matB,
                        &beta,
                        matC,
                        CUDA_R_32F,
                        CUSPARSE_CSRMM_ALG1,
                        &bufferSize) )

std::cout<<"bufferSize: "<<bufferSize<<std::endl;

  void *buffer = NULL;
  CHECK_CUDA( cudaMalloc(&buffer, bufferSize) )
  CHECK_CUSPARSE( cusparseSpMM(
                        handle.sparse,
                        CUSPARSE_OPERATION_NON_TRANSPOSE,
                        CUSPARSE_OPERATION_NON_TRANSPOSE,
                        &alpha,
                        matA,
                        matB,
                        &beta,
                        matC,
                        CUDA_R_32F,
                        CUSPARSE_CSRMM_ALG1,
                        buffer) )

  CHECK_CUSPARSE( cusparseDestroySpMat(matA) )
  CHECK_CUSPARSE( cusparseDestroyDnMat(matB) )
  CHECK_CUSPARSE( cusparseDestroyDnMat(matC) )
}
*/


void gemm_ssd_gpu(int m, int n, int k,
    int *hRowPtr, int *hColIdx, float *hVal, int nnz, float *hB, float *hC) {
  dvec<int> dRowPtr(hRowPtr, hRowPtr+m+1);
  dvec<int> dColIdx(hColIdx, hColIdx+nnz);
  dvec<float> dVal(hVal, hVal+nnz);
  dvec<float> dB(hB, hB+k*n);
  dvec<float> dC(m*n);
  GEMM_SDD(m, n, k, dRowPtr, dColIdx, dVal, nnz, dB, dC);
  thrust::copy(dC.begin(), dC.end(), hC);
}


// (sparse) A * (sparse) B = (dense) C
// customized sparse GEMM for the case where the csrRowPtrC and nnzC are known
void GEMM_SSD(int m, int n, int k, float alpha,
    int *csrRowPtrA, int *csrColIndA, float *csrValA, int nnzA,
    int *csrRowPtrB, int *csrColIndB, float *csrValB, int nnzB,
    int *csrRowPtrC, int *csrColIndC, float *csrValC, int nnzC,
    csrgemm2Info_t &info, cusparseHandle_t &handle, cusparseMatDescr_t &descr) {
  
  // dummy matrix D
  int nnzD = 0, *csrRowPtrD = 0, *csrColIndD = 0;
  float *csrValD = 0;
  
  size_t bufferSize;
  CHECK_CUSPARSE( cusparseScsrgemm2_bufferSizeExt(
        handle, m, n, k, &alpha,
        descr, nnzA, csrRowPtrA, csrColIndA,
        descr, nnzB, csrRowPtrB, csrColIndB,
        NULL,
        descr, nnzD, csrRowPtrD, csrColIndD,
        info,
        &bufferSize) )

  std::cout<<"[GEMM_SSD] buffersize: "<<bufferSize/1.e9<<" GB"<<std::endl;
  void *buffer = NULL;
  CHECK_CUDA( cudaMalloc(&buffer, bufferSize) )

  CHECK_CUSPARSE( cusparseScsrgemm2(handle, m, n, k, &alpha,
          descr, nnzA, csrValA, csrRowPtrA, csrColIndA,
          descr, nnzB, csrValB, csrRowPtrB, csrColIndB,
          NULL,
          descr, nnzD, csrValD, csrRowPtrD, csrColIndD,
          descr,       csrValC, csrRowPtrC, csrColIndC,
          info, buffer) )

  if (buffer != NULL) CHECK_CUDA( cudaFree(buffer) )
}


// (sparse) A * (sparse) B = (sparse) C
// csrRowPtrC, csrColIndC, csrValC and nnzC are allocated or computed 
void GEMM_SSS(int m, int n, int k, float alpha,
    int *csrRowPtrA, int *csrColIndA, float *csrValA, int nnzA,
    int *csrRowPtrB, int *csrColIndB, float *csrValB, int nnzB,
    int* &csrRowPtrC, int* &csrColIndC, float* &csrValC, int &nnzC,
    csrgemm2Info_t &info, cusparseHandle_t &handle, cusparseMatDescr_t &descr,
    float &t_gemm, float &t_nnz) {
  
  // dummy matrix D
  int nnzD = 0, *csrRowPtrD = 0, *csrColIndD = 0;
  float *csrValD = 0;
  
  size_t bufferSize;
  CHECK_CUSPARSE( cusparseScsrgemm2_bufferSizeExt(
        handle, m, n, k, &alpha,
        descr, nnzA, csrRowPtrA, csrColIndA,
        descr, nnzB, csrRowPtrB, csrColIndB,
        NULL,
        descr, nnzD, csrRowPtrD, csrColIndD,
        info,
        &bufferSize) )

  void *buffer = NULL;
  CHECK_CUDA( cudaMalloc(&buffer, bufferSize) )

  // step 3: compute csrRowPtrC
  TimerGPU t; t.start();
  CHECK_CUDA( cudaMalloc((void**)&csrRowPtrC, sizeof(int)*(m+1)) )
  int *nnzTotalDevHostPtr = &nnzC;
  cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST);
  
  CHECK_CUSPARSE( cusparseXcsrgemm2Nnz(handle, m, n, k,
          descr, nnzA, csrRowPtrA, csrColIndA,
          descr, nnzB, csrRowPtrB, csrColIndB,
          descr, nnzD, csrRowPtrD, csrColIndD,
          descr, csrRowPtrC, nnzTotalDevHostPtr,
          info, buffer) )
  if (NULL != nnzTotalDevHostPtr){
      nnzC = *nnzTotalDevHostPtr;
  }else{
      int baseC;
      cudaMemcpy(&nnzC, csrRowPtrC+m, sizeof(int), cudaMemcpyDeviceToHost);
      cudaMemcpy(&baseC, csrRowPtrC, sizeof(int), cudaMemcpyDeviceToHost);
      nnzC -= baseC;
  }  
  
  //std::cout<<"[GEMM_SSS] buffersize: "<<bufferSize/1.e9<<" GB"
  //         <<", C: "<<(m+1)/1.e9*4+nnzC/1.e9*4*2<<" GB"<<std::endl;

  // step 4: finish sparsity pattern and value of C
  // Remark: set csrValC to null if only sparsity pattern is required 
  CHECK_CUDA( cudaMalloc((void**)&csrColIndC, sizeof(int)*nnzC) )
  CHECK_CUDA( cudaMalloc((void**)&csrValC, sizeof(float)*nnzC) )
  t.stop(); t_nnz += t.elapsed_time();

  cudaProfilerStart();
  t.start();
  CHECK_CUSPARSE( cusparseScsrgemm2(handle, m, n, k, &alpha,
          descr, nnzA, csrValA, csrRowPtrA, csrColIndA,
          descr, nnzB, csrValB, csrRowPtrB, csrColIndB,
          NULL,
          descr, nnzD, csrValD, csrRowPtrD, csrColIndD,
          descr,       csrValC, csrRowPtrC, csrColIndC,
          info, buffer) )
  t.stop(); t_gemm += t.elapsed_time();
  cudaProfilerStop();

  // step 5: clean up
  if (buffer != NULL) CHECK_CUDA( cudaFree(buffer) )
}

