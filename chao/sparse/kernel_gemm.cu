#include "util_gpu.hpp"
#include "kernel_gemm.hpp"

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

  void *buffer = NULL;
  CHECK_CUDA( cudaMalloc(&buffer, bufferSize) )

  // step 3: compute csrRowPtrC
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
    //<<", C: "<<(m+1)/1.e9*4+nnzC/1.e9*4*2<<" GB"<<std::endl;

  // step 4: finish sparsity pattern and value of C
  // Remark: set csrValC to null if only sparsity pattern is required 
  CHECK_CUDA( cudaMalloc((void**)&csrColIndC, sizeof(int)*nnzC) )
  CHECK_CUDA( cudaMalloc((void**)&csrValC, sizeof(float)*nnzC) )
  CHECK_CUSPARSE( cusparseScsrgemm2(handle, m, n, k, &alpha,
          descr, nnzA, csrValA, csrRowPtrA, csrColIndA,
          descr, nnzB, csrValB, csrRowPtrB, csrColIndB,
          NULL,
          descr, nnzD, csrValD, csrRowPtrD, csrColIndD,
          descr,       csrValC, csrRowPtrC, csrColIndC,
          info, buffer) )

  // step 5: clean up
  if (buffer != NULL) CHECK_CUDA( cudaFree(buffer) )
}


