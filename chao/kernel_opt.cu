#include "timer.hpp"

#include <vector>
#include <algorithm> 

#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>

#include "cublas_v2.h"
#include <cuda_runtime.h>
#include "kernel_gpu.hpp"

#include "sort/sort_gpu.hpp"

template <typename T>
using dvec = thrust::device_vector<T>;


void compute_row_norms(dvec<float> &R, dvec<float> &Q, 
    dvec<float> &R2, dvec<float> &Q2, int nRow, int nCol) {

  dvec<int> Rrow(nRow), Qrow(nRow); // temporary storage for row indices
  auto zero = thrust::make_counting_iterator<int>(0);
  auto countItr = thrust::make_transform_iterator(zero, rowIdx(nCol));
  
  typedef thrust::device_vector<float>::iterator FloatIterator;
  thrust::transform_iterator<square, FloatIterator> R_iter(R.begin(), square());
  thrust::transform_iterator<square, FloatIterator> Q_iter(Q.begin(), square());
  thrust::reduce_by_key(countItr, countItr+nRow*nCol, R_iter, Rrow.begin(), R2.begin());
  thrust::reduce_by_key(countItr, countItr+nRow*nCol, Q_iter, Qrow.begin(), Q2.begin());
}


void compute_distance(const dvec<float> &R, const dvec<float> &Q, 
    const dvec<float> &R2, const dvec<float> &Q2, const dvec<float> &ones,
    dvec<float> &Dist, int nLeaf, int N, int d, int r, int m, cublasHandle_t &handle) {
  
  const float *ptrdR[nLeaf], *ptrdQ[nLeaf];
  float *ptrDist[nLeaf];
  for (int i=0; i<nLeaf; i++) {
    ptrdR[i] = thrust::raw_pointer_cast(R.data()+i*N*d);
    ptrdQ[i] = thrust::raw_pointer_cast(Q.data()+i*N*d+r*m*d);
    ptrDist[i] = thrust::raw_pointer_cast(Dist.data()+i*m*N);
  }

  // compute the distance (transpose) (cublas assumes column-major ordering)
  const float alpha = -2;
  const float beta = 0;
  const float one = 1;
  //cublasCheck( cublasSgemmBatched(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, m, d, &alpha,
    //      ptrdR, d, ptrdQ, d, &beta, ptrDist, N, nLeaf) );

  // TODO: use cublasSgemmBatchedStride()
  cublasCheck( cublasSgemmStridedBatched(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, m, d, &alpha,
          ptrdR[0], d, N*d, ptrdQ[0], d, N*d, &beta, ptrDist[0], N, m*N, nLeaf) );

  
  // rank-1 updates
  for (int i=0; i<nLeaf; i++) {
    cublasCheck( cublasSger(handle, N, m, &one, 
			      thrust::raw_pointer_cast(R2.data()+i*N), 1, 
			      thrust::raw_pointer_cast(ones.data()), 1,
			      ptrDist[i], N) );
    cublasCheck( cublasSger(handle, N, m, &one, 
			      thrust::raw_pointer_cast(ones.data()), 1, 
			      thrust::raw_pointer_cast(Q2.data()+i*N+r*m), 1,
			      ptrDist[i], N) );
  }
  
}


struct firstKCols : public thrust::unary_function<int, int> {
  int k, N;

  __host__ __device__
    firstKCols(int k_, int N_): k(k_), N(N_)  {}

  __host__ __device__
    int operator()(int i) {
      return i/k*N+i%k;
    }
};


struct strideBlock : public thrust::unary_function<int, int> {
  int mk, Nk, r;

  __host__ __device__
    strideBlock(int mk_, int Nk_, int r_): mk(mk_), Nk(Nk_), r(r_) {}

  __host__ __device__
    int operator()(int i) {
      return i/mk*Nk + r*mk + i%mk;
    }
};


void get_kcols(const dvec<float> &D, dvec<float> &Dk, int nLeaf, int m, int N, int k, int r) {
  auto zero  = thrust::make_counting_iterator<int>(0);
  auto iterD = thrust::make_transform_iterator(zero, firstKCols(k, N));
  auto permD = thrust::make_permutation_iterator(D.begin(), iterD);
  auto iterK = thrust::make_transform_iterator(zero, strideBlock(m*k, N*k, r));
  auto permK = thrust::make_permutation_iterator(Dk.begin(), iterK);
  thrust::copy(permD, permD+nLeaf*m*k, permK);
}


struct firstKVals : public thrust::unary_function<int, int> {
  int k, N, m;
  const int* vals;

  __host__ __device__
    firstKVals(int k_, int N_, int m_, const int *val_): 
      k(k_), N(N_), m(m_), vals(val_)  {}

  __host__ __device__
    int operator()(int i) {
      return vals[i/k*N+i%k]%N + i/(m*k)*N;
    }
};


void get_kcols(const dvec<int> &idx, const dvec<int> &ID,
    dvec<int> &IDk, int nLeaf, int m, int N, int k, int r) {
  const int* vals  = thrust::raw_pointer_cast(idx.data());
  auto zero  = thrust::make_counting_iterator<int>(0);
  auto iterD = thrust::make_transform_iterator(zero, firstKVals(k, N, m, vals));
  auto permD = thrust::make_permutation_iterator(ID.begin(), iterD);
  auto iterK = thrust::make_transform_iterator(zero, strideBlock(m*k, N*k, r));
  auto permK = thrust::make_permutation_iterator(IDk.begin(), iterK);
  thrust::copy(permD, permD+nLeaf*m*k, permK);
}


void find_knn(dvec<float> &Dist, const dvec<int> &ID, 
    dvec<float> &nborDist, dvec<int> &nborID, int nLeaf, int m, int N, int k, int r) {
  
  dvec<int> idx(m*nLeaf*N); // no need to initialize for mgpu
  sortGPU::sort_matrix_rows_mgpu(Dist, idx, m*nLeaf, N);
  get_kcols(Dist, nborDist, nLeaf, m, N, k, r);
  get_kcols(idx, ID, nborID, nLeaf, m, N, k, r);
}


void gemm_kselect_opt(int nLeaf, float *ptrR[], float *ptrQ[], int *ptrID[], int N, int d, 
		     float *ptrNborDist[], int *ptrNborID[], int k, int m,
         float &t_dist, float &t_sort, float &t_kernel) {

  // copy data to device
  dvec<float> R(N*d*nLeaf), Q(N*d*nLeaf);
  dvec<int>   ID(N*nLeaf); // ID of reference points
  for (int i=0; i<nLeaf; i++) {
    thrust::copy(ptrR[i], ptrR[i]+N*d, R.begin()+i*N*d);
    thrust::copy(ptrQ[i], ptrQ[i]+N*d, Q.begin()+i*N*d);
    thrust::copy(ptrID[i], ptrID[i]+N, ID.begin()+i*N);
  }
 
  // output
  dvec<float> nborDist(N*k*nLeaf);
  dvec<int>   nborID(N*k*nLeaf);

  // create CUBLAS handle
  cublasHandle_t handle;
  cublasCheck( cublasCreate(&handle) );  
  //sortGPU::init_mgpu();
  const dvec<float> ones(N, 1.0);


  Timer t, t1;
  cudaCheck( cudaDeviceSynchronize() ); t1.start();


  // compute row norms  
  dvec<float> R2(N*nLeaf), Q2(N*nLeaf);
  cudaCheck( cudaDeviceSynchronize() ); t.start();
  compute_row_norms(R, Q, R2, Q2, N*nLeaf, d);
  cudaCheck( cudaDeviceSynchronize() );
  t.stop(); t_dist += t.elapsed_time();


  // blocking
  assert(N%m==0); // m is block size
  int M = N/m; // number of blocks
  dvec<float> Dist(m*N*nLeaf); // block/partial results 

  for (int r=0; r<M; r++) {

    cudaCheck( cudaDeviceSynchronize() ); t.start();
    compute_distance(R, Q, R2, Q2, ones, Dist, nLeaf, N, d, r, m, handle);
    cudaCheck( cudaDeviceSynchronize() );
    t.stop(); t_dist += t.elapsed_time();


    cudaCheck( cudaDeviceSynchronize() ); t.start();
    find_knn(Dist, ID, nborDist, nborID, nLeaf, m, N, k, r);
    cudaCheck( cudaDeviceSynchronize() );
    t.stop(); t_sort += t.elapsed_time();
  }
  
  cudaCheck( cudaDeviceSynchronize() );
  t1.stop(); t_kernel += t1.elapsed_time();

  // copy results back to host
  for (int i=0; i<nLeaf; i++) {
    thrust::copy(nborDist.begin()+i*N*k, nborDist.begin()+(i+1)*N*k, ptrNborDist[i]);
    thrust::copy(nborID.begin()+i*N*k, nborID.begin()+(i+1)*N*k, ptrNborID[i]);
  }

  // clean up resouce
  cublasDestroy(handle);
  sortGPU::final_mgpu();
}

