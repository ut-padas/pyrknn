#include "util_gpu.hpp"
#include "print.hpp"
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
 

  //std::cout<<"k: "<<k<<", n: "<<n<<std::endl;
  //tprint(n,k,B,"dB");
  //dprint(m, k, nnzA, rowPtr, colIdx, val, "dA");

  cusparseDnMatDescr_t matB, matC;
  CHECK_CUSPARSE( cusparseCreateDnMat(&matB, k, n, k, valB, CUDA_R_32F, CUSPARSE_ORDER_COL) )
  CHECK_CUSPARSE( cusparseCreateDnMat(&matC, m, n, m, valC, CUDA_R_32F, CUSPARSE_ORDER_COL) )

  /*
  // for debugging
  int64_t rowB, colB, ldB;
  void *vB;
  cudaDataType dt;
  cusparseOrder_t od;
  CHECK_CUSPARSE( cusparseDnMatGet(matB, &rowB, &colB, &ldB, &vB, &dt, &od) )
  std::cout<<"row: "<<rowB<<", col: "<<colB<<", ldB: "<<ldB<<std::endl;
  */

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

  //std::cout<<"bufferSize: "<<bufferSize<<std::endl;

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


void gemm_sdd_gpu(int m, int n, int k,
    int *hRowPtr, int *hColIdx, float *hVal, int nnz, float *hB, float *hC) {
  dvec<int> dRowPtr(hRowPtr, hRowPtr+m+1);
  dvec<int> dColIdx(hColIdx, hColIdx+nnz);
  dvec<float> dVal(hVal, hVal+nnz);
  dvec<float> dB(hB, hB+k*n);
  dvec<float> dC(m*n);
  GEMM_SDD(m, n, k, dRowPtr, dColIdx, dVal, nnz, dB, dC);
  thrust::copy(dC.begin(), dC.end(), hC);
}


/*
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

  //std::cout<<"[GEMM_SSD] buffersize: "<<bufferSize/1.e9<<" GB"<<std::endl;
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
*/


// (sparse) A * (sparse) B = (sparse) C
// csrRowPtrC, csrColIndC, csrValC and nnzC are allocated or computed 
void GEMM_SSS(int m, int n, int k, float alpha,
    int *csrRowPtrA, int *csrColIndA, float *csrValA, int nnzA,
    int *csrRowPtrB, int *csrColIndB, float *csrValB, int nnzB,
    int* &csrRowPtrC, int* &csrColIndC, float* &csrValC, int &nnzC,
    float &t_gemm, float &t_nnz) {

    auto& HD = knnHandle_t::instance();
    cusparseHandle_t handle = HD.sparse;

    int A_num_rows = m, A_num_cols = k, A_nnz = nnzA;
    int B_num_rows = k, B_num_cols = n, B_nnz = nnzB;
    int C_num_rows = m;

    int   *dA_csrOffsets = csrRowPtrA, *dA_columns = csrColIndA;
    int   *dB_csrOffsets = csrRowPtrB, *dB_columns = csrColIndB;
    int   *&dC_csrOffsets = csrRowPtrC, *&dC_columns = csrColIndC;

    float *dA_values = csrValA, *dB_values = csrValB, *&dC_values = csrValC;

    // CUSPARSE APIs
    cusparseSpMatDescr_t matA, matB, matC;
    void*  dBuffer1    = NULL, *dBuffer2   = NULL;
    size_t bufferSize1 = 0,    bufferSize2 = 0;
    //CHECK_CUSPARSE( cusparseCreate(&handle) )
    // Create sparse matrix A in CSR format
    CHECK_CUSPARSE( cusparseCreateCsr(&matA, A_num_rows, A_num_cols, A_nnz,
                                      dA_csrOffsets, dA_columns, dA_values,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) )
    CHECK_CUSPARSE( cusparseCreateCsr(&matB, B_num_rows, B_num_cols, B_nnz,
                                      dB_csrOffsets, dB_columns, dB_values,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) )
    CHECK_CUSPARSE( cusparseCreateCsr(&matC, A_num_rows, B_num_cols, 0,
                                      NULL, NULL, NULL,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) )
    //--------------------------------------------------------------------------
    // SpGEMM Computation
    cusparseSpGEMMDescr_t spgemmDesc;
    CHECK_CUSPARSE( cusparseSpGEMM_createDescr(&spgemmDesc) )

    float               beta        = 0.0f;
    cusparseOperation_t opA         = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cusparseOperation_t opB         = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cudaDataType        computeType = CUDA_R_32F;

    // ask bufferSize1 bytes for external memory
    CHECK_CUSPARSE(
        cusparseSpGEMM_workEstimation(handle, opA, opB,
                                      &alpha, matA, matB, &beta, matC,
                                      computeType, CUSPARSE_SPGEMM_DEFAULT,
                                      spgemmDesc, &bufferSize1, NULL) )
    CHECK_CUDA( cudaMalloc((void**) &dBuffer1, bufferSize1) )
    // inspect the matrices A and B to understand the memory requirement for
    // the next step
    CHECK_CUSPARSE(
        cusparseSpGEMM_workEstimation(handle, opA, opB,
                                      &alpha, matA, matB, &beta, matC,
                                      computeType, CUSPARSE_SPGEMM_DEFAULT,
                                      spgemmDesc, &bufferSize1, dBuffer1) )

    // ask bufferSize2 bytes for external memory
    CHECK_CUSPARSE(
        cusparseSpGEMM_compute(handle, opA, opB,
                               &alpha, matA, matB, &beta, matC,
                               computeType, CUSPARSE_SPGEMM_DEFAULT,
                               spgemmDesc, &bufferSize2, NULL) )
    CHECK_CUDA( cudaMalloc((void**) &dBuffer2, bufferSize2) )

    // compute the intermediate product of A * B
    CHECK_CUSPARSE( cusparseSpGEMM_compute(handle, opA, opB,
                                           &alpha, matA, matB, &beta, matC,
                                           computeType, CUSPARSE_SPGEMM_DEFAULT,
                                           spgemmDesc, &bufferSize2, dBuffer2) )
    // get matrix C non-zero entries C_nnz1
    int64_t C_num_rows1, C_num_cols1, C_nnz1;
    CHECK_CUSPARSE( cusparseSpMatGetSize(matC, &C_num_rows1, &C_num_cols1,
                                         &C_nnz1) )
    // allocate matrix C
    nnzC = C_nnz1;
    CHECK_CUDA( cudaMalloc((void**) &dC_columns, C_nnz1 * sizeof(int))   )
    CHECK_CUDA( cudaMalloc((void**) &dC_values,  C_nnz1 * sizeof(float)) )
    CHECK_CUDA( cudaMalloc((void**) &dC_csrOffsets, (C_num_rows + 1) * sizeof(int)) )

    // update matC with the new pointers
    CHECK_CUSPARSE(
        cusparseCsrSetPointers(matC, dC_csrOffsets, dC_columns, dC_values) )

    // if beta != 0, cusparseSpGEMM_copy reuses/updates the values of dC_values

    // copy the final products to the matrix C
    CHECK_CUSPARSE(
        cusparseSpGEMM_copy(handle, opA, opB,
                            &alpha, matA, matB, &beta, matC,
                            computeType, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc) )

    // destroy matrix/vector descriptors
    CHECK_CUSPARSE( cusparseSpGEMM_destroyDescr(spgemmDesc) )
    CHECK_CUSPARSE( cusparseDestroySpMat(matA) )
    CHECK_CUSPARSE( cusparseDestroySpMat(matB) )
    CHECK_CUSPARSE( cusparseDestroySpMat(matC) )
}

void gemm_sss_gpu(int m, int n, int k, float alpha,
    int *rowPtrA, int *colIdxA, float *valA, int nnzA,
    int *rowPtrB, int *colIdxB, float *valB, int nnzB,
    int *&rowPtrC, int *&colIdxC, float *&valC, int &nnzC) {

  ivec dRowPtrA(rowPtrA, rowPtrA+m+1); 
  ivec dColIdxA(colIdxA, colIdxA+nnzA); 
  fvec dValA(valA, valA+nnzA);
  
  ivec dRowPtrB(rowPtrB, rowPtrB+k+1); 
  ivec dColIdxB(colIdxB, colIdxB+nnzB); 
  fvec dValB(valB, valB+nnzB);
  
  int *dRowPtrC; 
  int *dColIdxC; 
  float *dValC;

  float t_gemm, t_nnz;

  //std::cout<<"Before GEMM_SSS"<<std::endl;
  GEMM_SSS(m, n, k, alpha,
      thrust::raw_pointer_cast(dRowPtrA.data()), 
      thrust::raw_pointer_cast(dColIdxA.data()), 
      thrust::raw_pointer_cast(dValA.data()), 
      nnzA,
      thrust::raw_pointer_cast(dRowPtrB.data()), 
      thrust::raw_pointer_cast(dColIdxB.data()), 
      thrust::raw_pointer_cast(dValB.data()), 
      nnzB,
      dRowPtrC,
      dColIdxC,
      dValC,
      nnzC,
      t_gemm, t_nnz);
  //std::cout<<"After GEMM_SSS"<<std::endl;

  assert(nnzC > 0);
  rowPtrC = new int[m+1]; assert(rowPtrC!=NULL);
  colIdxC = new int[nnzC]; assert(colIdxC!=NULL);
  valC = new float[nnzC]; assert(valC!=NULL);

  CHECK_CUDA( cudaMemcpy(rowPtrC, dRowPtrC, (m+1)*sizeof(int), cudaMemcpyDeviceToHost) );
  CHECK_CUDA( cudaMemcpy(colIdxC, dColIdxC, nnzC*sizeof(int), cudaMemcpyDeviceToHost) );
  CHECK_CUDA( cudaMemcpy(valC, dValC, nnzC*sizeof(float), cudaMemcpyDeviceToHost) );
}


