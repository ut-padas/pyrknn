#ifndef KERNEL_GEMM_HPP
#define KERNEL_GEMM_HPP

#include <cusparse.h> 
#include "util_gpu.hpp"

void GEMM(int, int, int, const fvec&, const fvec&, fvec&);

void GEMM_SDD(int, int, int, ivec&, ivec&, fvec&, int, fvec&, fvec&);

void GEMM_SSD(int m, int n, int k, float alpha,
    int *csrRowPtrA, int *csrColIndA, float *csrValA, int nnzA,
    int *csrRowPtrB, int *csrColIndB, float *csrValB, int nnzB,
    int *csrRowPtrC, int *csrColIndC, float *csrValC, int nnzC,
    csrgemm2Info_t &info, cusparseHandle_t &handle, cusparseMatDescr_t &descr);


void GEMM_SSS(int m, int n, int k, float alpha,
    int *csrRowPtrA, int *csrColIndA, float *csrValA, int nnzA,
    int *csrRowPtrB, int *csrColIndB, float *csrValB, int nnzB,
    int* &csrRowPtrC, int* &csrColIndC, float* &csrValC, int &nnzC,
    csrgemm2Info_t &info, cusparseHandle_t &handle, cusparseMatDescr_t &descr,
    float&, float&);

#endif
