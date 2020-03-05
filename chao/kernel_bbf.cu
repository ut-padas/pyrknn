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

template <typename T>
using dvec = thrust::device_vector<T>;


void bb_gemm_kselect(int nLeaf, float *ptrR[], float *ptrQ[], int *ptrID[], int N, int d, 
		     float *ptrNborDist[], int *ptrNborID[], int k, 
         float &t_dist, float &t_sort, float &t_store, float &t_kernel) {

  // initialize data on device
  std::vector<dvec<float>> vecR, vecQ, vecNborDist;
  std::vector<dvec<int>> vecID, vecNborID;
  //std::vector<dve<int>> vecIdx; // results of k-select
  cudaStream_t streams[nLeaf];
  for (int i=0; i<nLeaf; i++) {
    vecR.push_back( dvec<float>(ptrR[i], ptrR[i]+N*d) );
    vecQ.push_back( dvec<float>(ptrQ[i], ptrQ[i]+N*d) );
    vecID.push_back( dvec<int>(ptrID[i], ptrID[i]+N) );
    vecNborDist.push_back( dvec<float>(N*k) );
    vecNborID.push_back( dvec<int>(N*k) );
    //vecIdx.push_back( dvec<int>(N*k) );
    cudaCheck( cudaStreamCreate(&streams[i]) );
  }
  // create a handle for CUBLAS
  cublasHandle_t handle;
  cublasCreate(&handle);  
  const float alpha = -2;
  const float beta = 0;
  const float one = 1;

  Timer t, t1;
  cudaDeviceSynchronize();
  t1.start();

  // compute row norms
  std::vector<dvec<float>> vecR2(nLeaf), vecQ2(nLeaf), vecRrow(nLeaf), vecQrow(nLeaf);
  dvec<float> ones(N, 1.0);
  typedef thrust::device_vector<float>::iterator FloatIterator;
  auto countItr = thrust::make_transform_iterator(thrust::counting_iterator<int>(0), 
	 	  				  linear_index_to_row_index<int>(d));
  for (int i=0; i<nLeaf; i++) {
    vecR2[i].resize(N);
    vecQ2[i].resize(N);
    vecRrow[i].resize(N);
    vecQrow[i].resize(N);
  } 
  
  cudaDeviceSynchronize();
  t.start();
  
  for (int i=0; i<nLeaf; i++) {
    thrust::transform_iterator<square, FloatIterator> R_iter(vecR[i].begin(), square());
    thrust::transform_iterator<square, FloatIterator> Q_iter(vecQ[i].begin(), square());
    //thrust::reduce_by_key(thrust::cuda::par.on(str[i]), 
    thrust::reduce_by_key(
		    countItr, countItr+N*d, R_iter, vecRrow[i].begin(), vecR2[i].begin());
    //thrust::reduce_by_key(thrust::cuda::par.on(str[i]), 
    thrust::reduce_by_key(
		    countItr, countItr+N*d, Q_iter, vecQrow[i].begin(), vecQ2[i].begin());
  }

  cudaDeviceSynchronize();
  t.stop(); t_dist += t.elapsed_time();

  // GEMM
  assert(N%d==0);
  int M = N/d; // number of blocks
  std::vector<dvec<float>> vecDist(nLeaf); // block/partial results 
  float *ptrdR[nLeaf], *ptrDist[nLeaf];
  for (int i=0; i<nLeaf; i++) {
    ptrdR[i] = thrust::raw_pointer_cast(vecR[i].data());
    vecDist[i].resize(d*N);
    ptrDist[i] = thrust::raw_pointer_cast(vecDist[i].data());
  }
  // blocking
  for (int r=0; r<M; r++) {
    float *ptrdQ[nLeaf];
    for (int i=0; i<nLeaf; i++) {
      ptrdQ[i] = thrust::raw_pointer_cast(vecQ[i].data()+r*d*d);
    }

    cudaDeviceSynchronize();
    t.start();
    // compute the distance (transpose) (cublas assumes column-major ordering)
    cublasCheck( cublasSgemmBatched(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, d, d, &alpha,
          ptrdR, d, ptrdQ, d, &beta, ptrDist, N, nLeaf) );

    // rank-1 updates
    for (int i=0; i<nLeaf; i++) {
      //cublasCheck( cublasSetStream(handle, streams[i]) );
      cublasCheck( cublasSger(handle, N, d, &one, 
			      thrust::raw_pointer_cast(vecR2[i].data()), 1, 
			      thrust::raw_pointer_cast(ones.data()), 1,
			      ptrDist[i], N) );
      cublasCheck( cublasSger(handle, N, d, &one, 
			      thrust::raw_pointer_cast(ones.data()), 1, 
			      thrust::raw_pointer_cast(vecQ2[i].data()+r*d), 1,
			      ptrDist[i], N) );
    }
    cudaDeviceSynchronize();
    t.stop(); t_dist += t.elapsed_time();


    // batched k-select
    dvec<float> D2(nLeaf*d*N);
    dvec<int> colIdx(nLeaf*d*N);
    dvec<int> rowIdx(nLeaf*d*N);
    for (int i=0; i<nLeaf; i++) {
      thrust::copy(vecDist[i].begin(), vecDist[i].end(), D2.begin()+i*d*N);
      for (int q=0; q<d; q++) {
        thrust::sequence(colIdx.begin()+i*d*N+q*N, colIdx.begin()+i*d*N+(q+1)*N, 0);
        thrust::fill_n(rowIdx.begin()+i*d*N+q*N, N, i*d+q);
      }
    }

    auto D2cpy = D2;
    
    cudaDeviceSynchronize();
    t.start();

    thrust::stable_sort_by_key(
          D2cpy.begin(), D2cpy.end(), colIdx.begin());
      
    thrust::stable_sort_by_key(
          D2.begin(), D2.end(), rowIdx.begin());
      
    thrust::stable_sort_by_key(
          rowIdx.begin(), rowIdx.end(), colIdx.begin());

    cudaDeviceSynchronize();
    t.stop(); t_sort += t.elapsed_time();
    
    cudaDeviceSynchronize();
    t.start();
    
    for (int i=0; i<nLeaf; i++) {
      // store neighbor ID
      auto iter = thrust::make_transform_iterator(thrust::counting_iterator<int>(0), 
                          modID<int>(k, N, thrust::raw_pointer_cast(colIdx.data()+i*d*N)));
      auto permID = thrust::make_permutation_iterator(vecID[i].begin(), iter);
      thrust::copy(permID, permID+d*k, vecNborID[i].begin()+r*d*k);
      // store neighbor distance
      auto iter2 = thrust::make_transform_iterator(thrust::counting_iterator<int>(0), 
                          modDist(k, N, thrust::raw_pointer_cast(colIdx.data()+i*d*N)));
      auto permDist = thrust::make_permutation_iterator(vecDist[i].begin(), iter2);
      thrust::copy(permDist, permDist+d*k, vecNborDist[i].begin()+r*d*k);
    }
    cudaDeviceSynchronize();
    t.stop(); t_store += t.elapsed_time();
  }
  
  cudaDeviceSynchronize();
  t1.stop(); t_kernel += t1.elapsed_time();

  // copy results back to host
  for (int i=0; i<nLeaf; i++) {
    cudaStreamSynchronize(streams[i]);
    cudaStreamDestroy(streams[i]);
    thrust::copy(vecNborDist[i].begin(), vecNborDist[i].end(), ptrNborDist[i]);
    thrust::copy(vecNborID[i].begin(), vecNborID[i].end(), ptrNborID[i]);
  }
  cublasDestroy(handle);
}

