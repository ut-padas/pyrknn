#include "util/timer.hpp"
#include "sort/sort_gpu.hpp"
#include "knn_gpu.hpp"

#include <vector>
#include <algorithm> 

#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>

#include "cublas_v2.h"
#include <cuda_runtime.h>

#define cudaCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

static const char *cudaGetErrorEnum(cublasStatus_t error) {
    switch (error) {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";

        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";

        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";

        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";

        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";

        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";

        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";

        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";
    }
    return "<unknown>";
}

#define cublasCheck(ans) { cublasAssert((ans), __FILE__, __LINE__); }
inline void cublasAssert(cublasStatus_t code, const char *file, int line, bool abort=true) {
   if (code != CUBLAS_STATUS_SUCCESS) {
      fprintf(stderr,"CUBLAS assert: %s %s %d\n", cudaGetErrorEnum(code), file, line);
      if (abort) exit(code);
   }
}

template <typename T>
using dvec = thrust::device_vector<T>;

struct square: public thrust::unary_function<float,float>
{
  __host__ __device__
  float operator()(float x) const
  {
    return x*x;
  }
};

void compute_row_norms(const float *R, const float *Q, 
    dvec<float> &R2, dvec<float> &Q2, int nRow, int nCol) {

  dvec<int> Rrow(nRow), Qrow(nRow); // temporary storage for row indices
  auto zero = thrust::make_counting_iterator<int>(0);
  auto countItr = thrust::make_transform_iterator(zero, rowIdx(nCol));
 
  typedef thrust::device_ptr<const float> dptr;
  dptr dR(R), dQ(Q);
  thrust::transform_iterator<square, dptr> R_iter(dR, square());
  thrust::transform_iterator<square, dptr> Q_iter(dQ, square());
  thrust::reduce_by_key(countItr, countItr+nRow*nCol, R_iter, Rrow.begin(), R2.begin());
  thrust::reduce_by_key(countItr, countItr+nRow*nCol, Q_iter, Qrow.begin(), Q2.begin());
}


void compute_distance(const float *R, const float *Q, 
    const dvec<float> &R2, const dvec<float> &Q2, const dvec<float> &ones,
    dvec<float> &Dist, int nLeaf, int N, int d, int r, int m, cublasHandle_t &handle) {
  
  float *ptrDist = thrust::raw_pointer_cast(Dist.data());

  // compute the distance (transpose) (cublas assumes column-major ordering)
  const float alpha = -2;
  const float beta = 0;
  const float one = 1;
  cublasCheck( cublasSgemmStridedBatched(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, m, d, &alpha,
          R, d, N*d, Q+r*m*d, d, N*d, &beta, ptrDist, N, m*N, nLeaf) );

  
  // rank-1 updates
  const int oneInt = 1;
  const float *ptrR2 = thrust::raw_pointer_cast(R2.data()),
      *ptrQ2 = thrust::raw_pointer_cast(Q2.data()),
      *ptrOne = thrust::raw_pointer_cast(ones.data());

  cublasCheck( cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_T, N, m, oneInt, &one, 
        ptrR2, N, N,
        ptrOne, m, m,
        &one, ptrDist, N, m*N, nLeaf) );
  
  cublasCheck( cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_T, N, m, oneInt, &one, 
        ptrOne, N, N,
        ptrQ2+r*m, m, N,
        &one, ptrDist, N, m*N, nLeaf) );
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


void get_kcols(const dvec<float> &D, float *Dk, int nLeaf, int m, int N, int k, int r) {
  auto zero  = thrust::make_counting_iterator<int>(0);
  auto iterD = thrust::make_transform_iterator(zero, firstKCols(k, N));
  auto permD = thrust::make_permutation_iterator(D.begin(), iterD);
  auto iterK = thrust::make_transform_iterator(zero, strideBlock(m*k, N*k, r));
  auto permK = thrust::make_permutation_iterator(thrust::device_ptr<float>(Dk), iterK);
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


void get_kcols(const dvec<int> &idx, const int *ID,
    int *IDk, int nLeaf, int m, int N, int k, int r) {
  const int* vals  = thrust::raw_pointer_cast(idx.data());
  auto zero  = thrust::make_counting_iterator<int>(0);
  auto iterD = thrust::make_transform_iterator(zero, firstKVals(k, N, m, vals));
  auto permD = thrust::make_permutation_iterator(thrust::device_ptr<const int>(ID), iterD);
  auto iterK = thrust::make_transform_iterator(zero, strideBlock(m*k, N*k, r));
  auto permK = thrust::make_permutation_iterator(thrust::device_ptr<int>(IDk), iterK);
  thrust::copy(permD, permD+nLeaf*m*k, permK);
}


void find_knn(dvec<float> &Dist, const int *ID, 
    float *nborDist, int *nborID, int nLeaf, int m, int N, int k, int r) {
  
  dvec<int> idx(m*nLeaf*N); // no need to initialize for mgpu
  sortGPU::sort_matrix_rows_mgpu(Dist, idx, m*nLeaf, N);
  get_kcols(Dist, nborDist, nLeaf, m, N, k, r);
  get_kcols(idx, ID, nborID, nLeaf, m, N, k, r);
}


void knn_gpu(const float *R, const float *Q, const int *ID, float *nborDist, int *nborID,
    int nLeaf, int N, int d, int k, int m, cublasHandle_t &handle,
    float &t_dist, float &t_sort, float &t_kernel) {

  const dvec<float> ones(N*nLeaf, 1.0);

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

  // initialize for CUBLAS and MGPU
  cublasHandle_t handle;
  cublasCheck( cublasCreate(&handle) );  
  sortGPU::init_mgpu();
 

  // run kernel
  knn_gpu(
      thrust::raw_pointer_cast(R.data()),
      thrust::raw_pointer_cast(Q.data()),
      thrust::raw_pointer_cast(ID.data()),
      thrust::raw_pointer_cast(nborDist.data()),
      thrust::raw_pointer_cast(nborID.data()),
      nLeaf, N, d, k, m, handle,
      t_dist, t_sort, t_kernel);


  // copy results back to host
  for (int i=0; i<nLeaf; i++) {
    thrust::copy(nborDist.begin()+i*N*k, nborDist.begin()+(i+1)*N*k, ptrNborDist[i]);
    thrust::copy(nborID.begin()+i*N*k, nborID.begin()+(i+1)*N*k, ptrNborID[i]);
  }

  // clean up resouce
  cublasDestroy(handle);
  sortGPU::final_mgpu();
}

