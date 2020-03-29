#include "util_gpu.hpp"
#include "kernel_gemm.hpp"
#include "knn_handle.hpp"
#include "timer_gpu.hpp"

#include <moderngpu/kernel_segsort.hxx>


struct average: public thrust::binary_function<int, int, int> {

  __host__ __device__
  average() {}

  __host__ __device__
  int operator()(int a, int b) {
    return (a+b)/2;
  }
};


struct shiftColIdx: public thrust::binary_function<int, int, int> {
  int d;

  __host__ __device__
  shiftColIdx(int d_): d(d_) {}

  __host__ __device__
  int operator()(int idx, int node) {
    return idx%d + d*node;
  }
};


void permute_sparse_matrix(int m, int n, int nnzA, 
    int *rowPtrA, int *colIdxA, float *valA, dvec<int> &perm,
    int* &rowPtrB, int* &colIdxB, float* &valB, int &nnzB,
    csrgemm2Info_t &info, cusparseHandle_t &handle, cusparseMatDescr_t &descr) {
  
  // create sparse permutation matrix
  dvec<float> ones(m, 1.0);
  dvec<int> rowPtrP(m+1);
  thrust::sequence(rowPtrP.begin(), rowPtrP.end(), 0);

  int *P_rowPtr = thrust::raw_pointer_cast(rowPtrP.data());
  int *P_colIdx = thrust::raw_pointer_cast(perm.data());
  float *P_val  = thrust::raw_pointer_cast(ones.data());
  GEMM_SSS(m, n, m, 1.0,
      P_rowPtr, P_colIdx, P_val, m,
      rowPtrA, colIdxA, valA, nnzA,
      rowPtrB, colIdxB, valB, nnzB,
      info, handle, descr);
}


// input sparse matrix is modified inplace
// assume n = d * nSegment
// seghead: start position of next level
void create_matrix_next_level(int *rowPtrA, int *colIdxA, float *valA, int m, int n, int nnzA,
    dvec<int> &perm, int *seghead, int nSegment, int d,
    csrgemm2Info_t &info, cusparseHandle_t &handle, cusparseMatDescr_t &descr) {
  
  // compute B = Perm * A
  int *rowPtrB, *colIdxB, nnzB;
  float *valB;
 
  //TimerGPU t; t.start();
  permute_sparse_matrix(m, n, nnzA, rowPtrA, colIdxA, valA,
      perm, rowPtrB, colIdxB, valB, nnzB,
      info, handle, descr);
  assert(nnzA == nnzB);
  //t.stop();
  //std::cout<<"permute sparse matrix: "<<t.elapsed_time()<<std::endl;

  // computing the node index
  // this is the bottleneck (90%)
  //t.start();
  dvec<int> shift(nnzB);
  thrust::counting_iterator<int> zero(0);
  auto cum_nnz = thrust::make_permutation_iterator(dptr<int>(rowPtrB), dptr<int>(seghead));
  thrust::upper_bound(cum_nnz+1, cum_nnz+nSegment+1, zero, zero+nnzB, shift.begin());
  //t.stop();
  //std::cout<<"compute node index: "<<t.elapsed_time()<<std::endl;


  //dprint(m+1, rowPtrB, "row pointer");
  //dprint(nSegment+1, seghead, "segment head");
  //print(shift, "node");

  // shift column indices
  //t.start();
  dptr<int> dColIdx(colIdxB);
  thrust::transform(dColIdx, dColIdx+nnzB, shift.begin(), dColIdx, shiftColIdx(d));
  //t.stop();
  //std::cout<<"shift column: "<<t.elapsed_time()<<std::endl;


  // overwrite results to input
  thrust::copy_n(thrust::device, dptr<int>(rowPtrB), m+1, dptr<int>(rowPtrA));
  thrust::copy_n(thrust::device, dptr<int>(colIdxB), nnzB, dptr<int>(colIdxA));
  thrust::copy_n(thrust::device, dptr<float>(valB), nnzB, dptr<float>(valA));


  // free allocation from calling GEMM_SSS
  CHECK_CUDA( cudaFree(rowPtrB) )
  CHECK_CUDA( cudaFree(colIdxB) )
  CHECK_CUDA( cudaFree(valB) )
}


// *** Input ***
// n = sum(N, nNode): total number of points
// N: number of points in every node
// valX[d*nNode]: assume random projections/vectors are given
// seghead[nNode+1]: start position of all segments/clusters
// segHeadNext: start position at next level
// *** Output ***
// median
// CSR format of the block sparse diagonal matrix
// permuted ID
void create_tree_next_level(int *ID, int *rowPtrP, int *colIdxP, float *valP, 
    int n, int d, int nnz, int *seghead, int *segHeadNext, int nNode,
    float *valX, float *median, 
    float &t_gemm, float &t_sort, float &t_mat) {

  //Access singleton
  auto const& handle = knnHandle_t::instance();

  csrgemm2Info_t info = handle.info;
  cusparseHandle_t hCusparse = handle.hCusparse;
  cusparseMatDescr_t descr = handle.descr;
  mgpu::standard_context_t &ctx = *(handle.ctx);
  
  // block diagonal for X
  dvec<int> rowPtrX(d*nNode+1);
  dvec<int> colIdxX(d*nNode);
  
  thrust::sequence(rowPtrX.begin(), rowPtrX.end(), 0); // one nonzero per row
  thrust::counting_iterator<int> zero(0);
  thrust::constant_iterator<int> DIM(d);
  thrust::transform(zero, zero+d*nNode, DIM, colIdxX.begin(), thrust::divides<int>());

  // block diagonal for Y = P * X
  dvec<int> rowPtrY(n+1);
  dvec<int> colIdxY(n);
  dvec<float> valY(n);
  thrust::sequence(rowPtrY.begin(), rowPtrY.end(), 0); // one nonzero per row

  // compute projections
  int *X_rowPtr = thrust::raw_pointer_cast(rowPtrX.data());
  int *X_colIdx = thrust::raw_pointer_cast(colIdxX.data());
  int *Y_rowPtr = thrust::raw_pointer_cast(rowPtrY.data());
  int *Y_colIdx = thrust::raw_pointer_cast(colIdxY.data());
  float *Y_val  = thrust::raw_pointer_cast(valY.data());

  //dprint(n, d*nNode, nnz, rowPtrP, colIdxP, valP, "P");
  //dprint(d*nNode, nNode, d*nNode, X_rowPtr, X_colIdx, valX, "X");

  std::cout<<"[Create level] X: "<<d/1.e9*nNode*4*2<<" GB"
    <<", Y: "<<3*4*n/1.e9<<" GB\n";

  TimerGPU t; t.start();
  GEMM_SSD(n, nNode, d*nNode, 1.0,
      rowPtrP, colIdxP, valP, nnz,
      X_rowPtr, X_colIdx, valX, d*nNode,
      Y_rowPtr, Y_colIdx, Y_val, n,
      info, hCusparse, descr);
  t.stop(); t_gemm += t.elapsed_time();
  //print(valY, "projection");


  // sort
  t.start();
  dvec<int> idx(n); //thrust::sequence(idx.begin(), idx.end(), 0);
  int *idxPtr = thrust::raw_pointer_cast(idx.data());
  mgpu::segmented_sort_indices(Y_val, idxPtr, n, seghead, nNode, mgpu::less_t<float>(), ctx);
  t.stop(); t_sort += t.elapsed_time();

  //print(idx, "index");

  // permute ID
  dvec<int> IDcpy(dptr<int>(ID), dptr<int>(ID)+n);
  auto permID = thrust::make_permutation_iterator(IDcpy.begin(), idx.begin());
  thrust::copy(thrust::device, permID, permID+n, ID);
  
  //dprint(n, ID, "permuted ID");
  
  // get median
  dvec<int> medpos(nNode); // index of median position
  dptr<int> segPtr(seghead);
  thrust::transform(segPtr, segPtr+nNode, segPtr+1, medpos.begin(), average());
  auto perm = thrust::make_permutation_iterator(valY.begin(), medpos.begin());
  thrust::copy(thrust::device, perm, perm+nNode, median);
  
  //dprint(nNode, median, "median");

  // create block diagonal matrix for next level
  t.start();
  create_matrix_next_level(rowPtrP, colIdxP, valP, n, d*nNode, nnz, 
      idx, segHeadNext, 2*nNode, d, 
      info, hCusparse, descr);
  t.stop(); t_mat += t.elapsed_time();

  //dprint(n, 2*d*nNode, nnz, rowPtrP, colIdxP, valP, "P next");
}



