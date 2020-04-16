#include "util_gpu.hpp"
#include "op_gpu.hpp"
#include "knn_handle.hpp"
#include "sort.hpp"


void compute_row_norms(const fvec &P, fvec &P_norm, int m, int n) {
  ivec P_row(m);
  auto zero = thrust::make_counting_iterator<int>(0);
  auto iter = thrust::make_transform_iterator(zero, rowIdx(n));
  auto P_square = thrust::make_transform_iterator(P.begin(), thrust::square<float>());
  thrust::reduce_by_key(iter, iter+m*n, P_square, P_row.begin(), P_norm.begin());
}


void compute_distance(const fvec &P, const fvec &normP, const fvec &ones, fvec &Dist,
    int nLeaf, int N, int d, int blk, int m, int offset) {
  
  auto& handle = knnHandle_t::instance();
  const float alpha = -2;
  const float beta = 0;
  const float one = 1;
  const float *ptrP = thrust::raw_pointer_cast(P.data());
  float *ptrD = thrust::raw_pointer_cast(Dist.data());

  CHECK_CUBLAS( cublasSgemmStridedBatched(handle.blas, CUBLAS_OP_T, CUBLAS_OP_N,
        N, m, d, &alpha, ptrP, d, N*d, ptrP+offset*d, d, N*d, 
        &beta, ptrD, N, m*N, nLeaf) );

  // rank-1 updates
  const int oneInt = 1;
  const float *ptrNorm = thrust::raw_pointer_cast(normP.data());
  const float *ptrOne = thrust::raw_pointer_cast(ones.data());

  CHECK_CUBLAS( cublasSgemmStridedBatched(
        handle.blas, CUBLAS_OP_N, CUBLAS_OP_T, N, m, oneInt, &one, 
        ptrNorm, N, N,
        ptrOne, m, m,
        &one, ptrD, N, m*N, nLeaf) );
  
  CHECK_CUBLAS( cublasSgemmStridedBatched(
        handle.blas, CUBLAS_OP_N, CUBLAS_OP_T, N, m, oneInt, &one, 
        ptrOne, N, N,
        ptrNorm+offset, m, N,
        &one, ptrD, N, m*N, nLeaf) );
}


void get_kcols_dist(const dvec<float> &D, float *Dk,
    int nLeaf, int m, int LD, int k, int N, int offset) {
  auto zero  = thrust::make_counting_iterator<int>(0);
  auto iterD = thrust::make_transform_iterator(zero, firstKCols(k, N));
  auto permD = thrust::make_permutation_iterator(D.begin(), iterD);
  auto iterK = thrust::make_transform_iterator(zero, strideBlock(k, m, N, LD, offset*LD));
  auto permK = thrust::make_permutation_iterator(thrust::device_ptr<float>(Dk), iterK);
  thrust::copy(permD, permD+nLeaf*m*k, permK);
}


void get_kcols_ID(const dvec<int> &permIdx, int *IDk, const ivec &ID,
    int nLeaf, int m, int LD, int k, int N, int offset) {
  const int *pIdx  = thrust::raw_pointer_cast(permIdx.data());
  auto zero  = thrust::make_counting_iterator<int>(0);
  auto iterD = thrust::make_transform_iterator(zero, firstKVals(k, m, N, pIdx));
  auto permD = thrust::make_permutation_iterator(ID.begin(), iterD);
  auto iterK = thrust::make_transform_iterator(zero, strideBlock(k, m, N, LD, offset*LD));
  auto permK = thrust::make_permutation_iterator(thrust::device_ptr<int>(IDk), iterK);
  thrust::copy(permD, permD+nLeaf*m*k, permK);
}


void find_neighbors(fvec& Dist, const ivec &ID, float *nborDist, int *nborID, 
    int nLeaf, int m, int offset, int N, int k, int LD, int blk) {
  
  ivec idx(m*nLeaf*N);
  sort_matrix_rows_mgpu(Dist, idx, m*nLeaf, N);
  get_kcols_dist(Dist, nborDist, nLeaf, m, LD, k, N, offset);
  get_kcols_ID(idx, nborID, ID, nLeaf, m, LD, k, N, offset);
}


// N: # points in ONE leaf node
void leaf_knn(const ivec &ID, const fvec &P, int N, int d, int nLeaf,
    int *nborID, float *nborDist, int k, int LD, int blkPoint) {

  // auxilliary data for distance calculation
  const dvec<float> ones(N*nLeaf, 1.0);

  // compute row norms  
  dvec<float> Pnorm(N*nLeaf);

  compute_row_norms(P, Pnorm, N*nLeaf, d);

  // blocking of points
  dvec<float> Dist(blkPoint*N*nLeaf); // block/partial results 
  int nBlock = (N+blkPoint-1)/blkPoint;
  for (int i=0; i<nBlock; i++) {
    int offset  = i*blkPoint;
    int blkSize = std::min(blkPoint, N-offset);
    compute_distance(P, Pnorm, ones, Dist, nLeaf, N, d, i, blkSize, offset);
    find_neighbors(Dist, ID, nborDist, nborID, nLeaf, blkSize, offset, N, k, LD, i);
  }
}


