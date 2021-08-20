#include "reorder.hpp"
#include "knn_handle.hpp"
#include "gemm.hpp"
#include "print.hpp"

#include <thrust/gather.h>


void gather(dvec<int> &x, const dvec<int>& perm) {
  assert(x.size() == perm.size());
  auto copy = x;
  //tprint(x, "[gather] before");
  //tprint(perm, "[gather] map");
  thrust::gather(perm.begin(), perm.end(), copy.begin(), x.begin());
  //tprint(x, "[gather] after");
}


void gather(float *x, int n, const dvec<int>& perm) {
  assert(perm.size() == n);
  dptr<float> xptr(x);
  fvec copy(xptr, xptr+n);
  thrust::gather(perm.begin(), perm.end(), copy.begin(), xptr);
}


void gather(fvec &A, int m, int n, const ivec &order) {
  const int  *p = thrust::raw_pointer_cast(order.data());
  auto copy = A;
  auto zero = thrust::make_counting_iterator<int>(0);
  auto iter = thrust::make_transform_iterator(zero, permMatRow(p, n, n));
  auto perm = thrust::make_permutation_iterator(copy.begin(), iter);
  thrust::copy(perm, perm+m*n, A.begin());
}


void gather_gpu(float *hA, int m, int n, const int *hP) {
  fvec dA(hA, hA+m*n);
  ivec dP(hP, hP+m);
  gather(dA, m, n, dP);
  thrust::copy_n(dA.begin(), m*n, hA);
}


void gather(dvec<int> &A_rowPtr, dvec<int> &A_colIdx, dvec<float> &A_val, 
    int m, int n, int nnz, dvec<int> &perm) {
  assert(perm.size() == m);
  // create sparse permutation matrix
  dvec<float> ones(m, 1.0);
  dvec<int> rowPtrP(m+1);
  thrust::sequence(rowPtrP.begin(), rowPtrP.end(), 0);

  int *P_rowPtr = thrust::raw_pointer_cast(rowPtrP.data());
  int *P_colIdx = thrust::raw_pointer_cast(perm.data());
  float *P_val  = thrust::raw_pointer_cast(ones.data());

  int *rowPtrA = thrust::raw_pointer_cast(A_rowPtr.data());
  int *colIdxA = thrust::raw_pointer_cast(A_colIdx.data());
  float *valA  = thrust::raw_pointer_cast(A_val.data());

  int *rowPtrB, *colIdxB, nnzB;
  float *valB;

  float t0 = 0., t1 = 0.;
  GEMM_SSS(m, n, m, 1.0,
      P_rowPtr, P_colIdx, P_val, m,
      rowPtrA, colIdxA, valA, nnz,
      rowPtrB, colIdxB, valB, nnzB,
      t0, t1);

  assert(nnz == nnzB);

  thrust::copy_n(thrust::device, rowPtrB, m+1, A_rowPtr.begin());
  thrust::copy_n(thrust::device, colIdxB, nnz, A_colIdx.begin());
  thrust::copy_n(thrust::device, valB, nnz, A_val.begin());

  CHECK_CUDA( cudaFree(rowPtrB) )
  CHECK_CUDA( cudaFree(colIdxB) )
  CHECK_CUDA( cudaFree(valB) )
}

void gather_gpu(int *hRowPtr, int *hColIdx, float *hVal, int m, int n, int nnz, int *hP) {
  ivec dRowPtr(hRowPtr, hRowPtr+m+1);
  ivec dColIdx(hColIdx, hColIdx+nnz);
  fvec dVal(hVal, hVal+nnz);
  ivec dP(hP, hP+m);
  gather(dRowPtr, dColIdx, dVal, m, n, nnz, dP);
  thrust::copy_n(dRowPtr.begin(), m+1, hRowPtr);
  thrust::copy_n(dColIdx.begin(), nnz, hColIdx);
  thrust::copy_n(dVal.begin(), nnz, hVal);
}

void scatter_gpu(float *hX, int m, int n, int *hP) {
  fvec dX(hX, hX+m*n);
  ivec dP(hP, hP+m);
  scatter(dX, m, n, dP);
  thrust::copy_n(dX.begin(), m*n, hX);
}

