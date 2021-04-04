#include "util_gpu.hpp"
#include "op_gpu.hpp"
#include "knn_handle.hpp"
#include "sort.hpp"
#include "timer_gpu.hpp"


void compute_row_norms(const fvec &P, fvec &P_norm, int m, int n) {
  ivec P_row(m);
  auto zero = thrust::make_counting_iterator<int>(0);
  auto iter = thrust::make_transform_iterator(zero, rowIdx(n));
  auto P_square = thrust::make_transform_iterator(P.begin(), thrust::square<float>());
  thrust::reduce_by_key(iter, iter+m*n, P_square, P_row.begin(), P_norm.begin());
}


void compute_distance(const fvec &P, const fvec &normP, const fvec &ones, fvec &Dist,
    int nLeaf, int N, int d, int blk, int m, int offset, float &t_gemm, float &t_rank) {

  // -----
  // INPUT
  // -----
  // P: (permuted) coordinates of data points, (N*nLeaf)-by-d matrix in row major
  // normP: squared two norm of all data points, length N*nLeaf
  // ones: auxilliary array of one's, length N*nLeaf
  // nLeaf: # of leaf nodes
  // N: # of points in ONE leaf node
  // d: dimension/number of coordinates of a data point
  // blk: index of batch
  // m: # of points in a batch for the compute_distance kernel
  // offset: blk*m

  // ------
  // OUTPUT
  // ------
  // Dist: distances between a batch of points and all points in a leaf node (for all nodes), 
  //        (m*nLeaf)-by-N matrix in row major
  // t_gemm & t_rank: timings

  auto& handle = knnHandle_t::instance();
  const float alpha = -2;
  const float beta = 0;
  const float one = 1;
  const float *ptrP = thrust::raw_pointer_cast(P.data());
  float *ptrD = thrust::raw_pointer_cast(Dist.data());

  TimerGPU t; t.start();
  CHECK_CUBLAS( cublasSgemmStridedBatched(handle.blas, CUBLAS_OP_T, CUBLAS_OP_N,
        N, m, d, &alpha, ptrP, d, N*d, ptrP+offset*d, d, N*d, 
        &beta, ptrD, N, m*N, nLeaf) );
  t.stop(); t_gemm += t.elapsed_time();

  // rank-1 updates
  t.start();
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
  t.stop(); t_rank += t.elapsed_time();
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


void find_neighbors(fvec& Dist, const ivec &ID, ivec &idx, float *nborDist, int *nborID, 
    int nLeaf, int m, int offset, int N, int k, int LD, int blk, float &t_sort) {
  
  TimerGPU t; t.start();
  sort_matrix_rows_mgpu(Dist, idx, m*nLeaf, N);
  t.stop(); t_sort += t.elapsed_time();

  get_kcols_dist(Dist, nborDist, nLeaf, m, LD, k, N, offset);
  get_kcols_ID(idx, nborID, ID, nLeaf, m, LD, k, N, offset);
}


void leaf_knn(const ivec &ID, const fvec &P, int N, int d, int nLeaf,
    int *nborID, float *nborDist, int k, int LD, int blkPoint, 
    float &t_dist, float &t_sort) {

  // -----
  // INPUT
  // -----
  // ID: (permuted) ID of data points
  // P: (permuted) coordinates of data points
  // N: # of points in ONE leaf node
  // d: dimension/number of coordinates of a data point
  // nLeaf: # of leaf nodes
  // k: # of nearest neighbors to compute
  // LD: # of columns of the two output matrix 'nborID' and 'nborDist'
  // blkPoint: # of points in a batch for the compute_distance kernel

  // ------
  // OUTPUT
  // ------
  // nborID: IDs of KNN of all points
  // nborDist: distances of KNN of all points
  // t_dist & t_sort: timings


  float t_kernel = 0., t_gemm = 0., t_rank = 0., t_nbor = 0., t_norm = 0.;
  TimerGPU t0, t1; t0.start();

  t1.start();
  // auxilliary data for distance calculation
  const dvec<float> ones(N*nLeaf, 1.0);

  // compute row norms  
  dvec<float> Pnorm(N*nLeaf);
  
  // blocking of points
  dvec<float> Dist(blkPoint*N*nLeaf); // block/partial results 
  dvec<int> Idx(blkPoint*N*nLeaf);  // auxiliary memory for sorting
  t1.stop(); t_norm += t1.elapsed_time();

  compute_row_norms(P, Pnorm, N*nLeaf, d);

  int nBlock = (N+blkPoint-1)/blkPoint;
  for (int i=0; i<nBlock; i++) {
    int offset  = i*blkPoint;
    int blkSize = std::min(blkPoint, N-offset);
    
    t1.start();
    compute_distance(P, Pnorm, ones, Dist, nLeaf, N, d, i, blkSize, offset, t_gemm, t_rank);
    t1.stop(); t_dist += t1.elapsed_time();

    t1.start();
    find_neighbors(Dist, ID, Idx, nborDist, nborID, nLeaf, blkSize, offset, N, k, LD, i, t_sort);
    t1.stop(); t_nbor += t1.elapsed_time();
  }
  t0.stop(); t_kernel = t0.elapsed_time();
  
  /*
  printf("\n===========================");
  printf("\n    Leaf Kernel Timing");
  printf("\n---------------------------");
  printf("\n* Malloc: %.2e s (%.0f %%)", t_norm, 100.*t_norm/t_kernel);
  printf("\n* Distance: %.2e s (%.0f %%)", t_dist, 100.*t_dist/t_kernel);
  printf("\n  -  rank: %.2e s (%.0f %%)", t_rank, 100.*t_rank/t_dist);
  printf("\n  -  gemm: %.2e s (%.0f %%)", t_gemm, 100.*t_gemm/t_dist);
  printf("\n* Neighbors: %.2e s (%.0f %%)", t_nbor, 100.*t_nbor/t_kernel);
  printf("\n  -  sort: %.2e s (%.0f %%)", t_sort, 100.*t_sort/t_nbor);
  printf("\n===========================\n");
  */
}


