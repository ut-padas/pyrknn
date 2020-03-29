#include <cusparse.h> 

void GEMM_SSD(int m, int n, int k, float alpha,
    int *csrRowPtrA, int *csrColIndA, float *csrValA, int nnzA,
    int *csrRowPtrB, int *csrColIndB, float *csrValB, int nnzB,
    int *csrRowPtrC, int *csrColIndC, float *csrValC, int nnzC,
    csrgemm2Info_t &info, cusparseHandle_t &handle, cusparseMatDescr_t &descr);


void GEMM_SSS(int m, int n, int k, float alpha,
    int *csrRowPtrA, int *csrColIndA, float *csrValA, int nnzA,
    int *csrRowPtrB, int *csrColIndB, float *csrValB, int nnzB,
    int* &csrRowPtrC, int* &csrColIndC, float* &csrValC, int &nnzC,
    csrgemm2Info_t &info, cusparseHandle_t &handle, cusparseMatDescr_t &descr);


