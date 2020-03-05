#include "timer.hpp"

#include <vector>
#include <algorithm> 

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>

#include "cublas_v2.h"
#include <cuda_runtime.h>
#include "kernel_gpu.hpp"

template <typename T>
using dvec = thrust::device_vector<T>;


void distSquared_gpu_stream(int nLeaf, float *R[], float *Q[], float *D2[],
		            int *Nr, int *Nq, int d) {

  //print(Q[0], Nq[0]*d, "Q0");
  //print(Q[1], Nq[1]*d, "Q1");

  cudaStream_t str[nLeaf];
  std::vector<thrust::device_vector<float>> vecR;
  std::vector<thrust::device_vector<float>> vecQ;
  std::vector<thrust::device_vector<float>> vecD2(nLeaf);
  for (int i=0; i<nLeaf; i++) {
    cudaStreamCreate(&str[i]);
    vecR.push_back( thrust::device_vector<float>(std::vector<float>(R[i], R[i]+Nr[i]*d)) );
    vecQ.push_back( thrust::device_vector<float>(std::vector<float>(Q[i], Q[i]+Nq[i]*d)) );
    vecD2[i].resize(Nq[i]*Nr[i]);
  }

  // initialization for row norms
  std::vector<thrust::device_vector<float>> vecR2(nLeaf);
  std::vector<thrust::device_vector<float>> vecQ2(nLeaf);
  std::vector<thrust::device_vector<float>> vecRrow(nLeaf);
  std::vector<thrust::device_vector<float>> vecQrow(nLeaf);
  std::vector<thrust::device_vector<float>> vecOnes(nLeaf);
  typedef thrust::device_vector<float>::iterator FloatIterator;
  auto countItr = thrust::make_transform_iterator(thrust::counting_iterator<int>(0), 
	 	  				  linear_index_to_row_index<int>(d));
  for (int i=0; i<nLeaf; i++) {
    vecR2[i].resize(Nr[i]);
    vecQ2[i].resize(Nq[i]);
    vecRrow[i].resize(Nr[i]);
    vecQrow[i].resize(Nq[i]);
    vecOnes[i].resize(std::max(Nq[i], Nr[i]), 1.0);
    //thrust::device_vector<int>   Rrow(Nr[i]);
    //thrust::device_vector<int>   Qrow(Nq[i]);
  }
  
  // Create a handle for CUBLAS
  cublasHandle_t handle;
  cublasCreate(&handle);  
  const float alpha = -2;
  const float beta = 0;
  const float one = 1;

  double t_gpu = 0.;
  cudaDeviceSynchronize();
  Timer t; t.start();

  for (int i=0; i<nLeaf; i++) {
    thrust::transform_iterator<square, FloatIterator> R_iter(vecR[i].begin(), square());
    thrust::transform_iterator<square, FloatIterator> Q_iter(vecQ[i].begin(), square());
    // compute row sums by summing values with equal row indices
    thrust::reduce_by_key(thrust::cuda::par.on(str[i]), 
		    countItr, countItr+Nr[i]*d, R_iter, vecRrow[i].begin(), vecR2[i].begin());
    thrust::reduce_by_key(thrust::cuda::par.on(str[i]), 
		    countItr, countItr+Nq[i]*d, Q_iter, vecQrow[i].begin(), vecQ2[i].begin());
  }

  for (int i=0; i<nLeaf; i++) {
    cublasSetStream(handle, str[i]);
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, Nq[i], Nr[i], d, &alpha, 
	        thrust::raw_pointer_cast(vecQ[i].data()), d, 
	        thrust::raw_pointer_cast(vecR[i].data()), d, &beta, 
	        thrust::raw_pointer_cast(vecD2[i].data()), Nq[i]);
  }
  
  for (int i=0; i<nLeaf; i++) {
    cublasSetStream(handle, str[i]);
    // rank-one update    
    cublasSger(handle, Nq[i], Nr[i], &one, 
  	       thrust::raw_pointer_cast(vecQ2[i].data()), 1,
	       thrust::raw_pointer_cast(vecOnes[i].data()), 1,
	       thrust::raw_pointer_cast(vecD2[i].data()), Nq[i]);
    cublasSger(handle, Nq[i], Nr[i], &one, 
	       thrust::raw_pointer_cast(vecOnes[i].data()), 1,
	       thrust::raw_pointer_cast(vecR2[i].data()), 1,
	       thrust::raw_pointer_cast(vecD2[i].data()), Nq[i]);
  }
  
  cudaDeviceSynchronize();
  t.stop(); t_gpu += t.elapsed_time();
  std::cout<<"Compute distance on GPU: "<<t_gpu<<" s"<<std::endl;

  // Finish
  cublasDestroy(handle);
  for (int i=0; i<nLeaf; i++) {
    cudaStreamSynchronize(str[i]);
    cudaStreamDestroy(str[i]);
    thrust::copy(vecD2[i].begin(), vecD2[i].end(), D2[i]);
  }
}

void distSquared_gpu(const float *R, const float *Q, float* D2, int Nr, int Nq, int d) {
  
  thrust::device_vector<float> d_R(R, R+Nr*d);
  thrust::device_vector<float> d_Q(Q, Q+Nq*d);

  //print(d_R, "d_R");
  //print(d_Q, "d_Q");
  
  // allocate storage for row sums and indices
  thrust::device_vector<float> d_R2(Nr);
  thrust::device_vector<float> d_Q2(Nq);
  thrust::device_vector<int>   Rrow(Nr);
  thrust::device_vector<int>   Qrow(Nq);
  typedef thrust::device_vector<float>::iterator FloatIterator;
  thrust::transform_iterator<square, FloatIterator> d_R2_iter(d_R.begin(), square());
  thrust::transform_iterator<square, FloatIterator> d_Q2_iter(d_Q.begin(), square());

  auto iterRrow = thrust::make_transform_iterator(thrust::counting_iterator<int>(0), 
	 	  				  linear_index_to_row_index<int>(d));
  auto iterQrow = thrust::make_transform_iterator(thrust::counting_iterator<int>(0), 
	 	  				  linear_index_to_row_index<int>(d));

  // compute row sums by summing values with equal row indices
  thrust::reduce_by_key(iterRrow, iterRrow+Nr*d, d_R2_iter, Rrow.begin(), d_R2.begin());
  thrust::reduce_by_key(iterQrow, iterQrow+Nq*d, d_Q2_iter, Qrow.begin(), d_Q2.begin());
  
  //print(d_R2, "d_R2");
  //print(d_Q2, "d_Q2");

  // D = -2 Q*R^T
  thrust::device_vector<float> d_D2(Nq*Nr);

  const float alpha = -2;
  const float beta = 0;

  // Create a handle for CUBLAS
  cublasHandle_t handle;
  cublasCreate(&handle);
  
  // Do the actual multiplication
  cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, Nq, Nr, d, &alpha, 
	      thrust::raw_pointer_cast(&d_Q[0]), d, 
	      thrust::raw_pointer_cast(&d_R[0]), d, &beta, 
	      thrust::raw_pointer_cast(&d_D2[0]), Nq);
  
  //print(d_D2, "d_D2");

  const float one = 1;
  thrust::device_vector<float> ones(std::max(Nq,Nr), 1.0);
  cublasSger(handle, Nq, Nr, &one, 
	     thrust::raw_pointer_cast(&d_Q2[0]), 1,
	     thrust::raw_pointer_cast(&ones[0]), 1,
	     thrust::raw_pointer_cast(&d_D2[0]), Nq);

  cublasSger(handle, Nq, Nr, &one, 
	     thrust::raw_pointer_cast(&ones[0]), 1,
	     thrust::raw_pointer_cast(&d_R2[0]), 1,
	     thrust::raw_pointer_cast(&d_D2[0]), Nq);

  //print(d_D2, "d_D2");

  // Destroy the handle
  cublasDestroy(handle);

  thrust::copy(d_D2.begin(), d_D2.end(), D2);
}

void kselect_gpu_stream(int nLeaf, float *ptrD2[], int *ptrID[], int *Nq, int *Nr,
		        float *ptrNborDist[], int *ptrNborID[], int k) {
  // copy data to device
  std::vector<dvec<float>> vecD2;
  std::vector<dvec<int>> vecID;
  std::vector<dvec<float>> vecNborDist(nLeaf); // result
  std::vector<dvec<int>> vecNborID(nLeaf); // result
  cudaStream_t str[nLeaf];
  for (int i=0; i<nLeaf; i++) {
    vecD2.push_back( dvec<float>(ptrD2[i], ptrD2[i]+Nq[i]*Nr[i]) );
    vecID.push_back( dvec<int>(ptrID[i], ptrID[i]+Nr[i]) );
    vecNborDist[i].resize(Nq[i]*k);
    vecNborID[i].resize(Nq[i]*k);
    cudaStreamCreate(&str[i]);
  }
  // find neighbors
  for (int i=0; i<nLeaf; i++) {
    dvec<int> idx(Nr[i]);
    for (int q=0; q<Nq[i]; q++) {
      thrust::sequence(idx.begin(), idx.end(), 0);  // initialize indices
      auto value = vecD2[i].data() + Nr[i]*q;
      thrust::stable_sort(thrust::cuda::par.on(str[i]),
		          idx.begin(), idx.end(), 
		          compare<float>(thrust::raw_pointer_cast(value)));
      for (int j=0; j<k; j++) {
        vecNborDist[i][j+q*k] = vecD2[i][ idx[j]+q*Nr[i] ];
	vecNborID[i][j+q*k] = vecID[i][ idx[j] ];
      } 
    }
  }
  // copy data to host
  for (int i=0; i<nLeaf; i++) {
    cudaStreamSynchronize(str[i]);
    cudaStreamDestroy(str[i]);
    thrust::copy(vecNborDist[i].begin(), vecNborDist[i].end(), ptrNborDist[i]);
    thrust::copy(vecNborID[i].begin(), vecNborID[i].end(), ptrNborID[i]);
  }
}

void kselect_gpu(const float *value, const int *ID, int n, float *kval, int *kID, int k) {
  thrust::device_vector<float> d_value(value, value+n);
  thrust::device_vector<int> d_ID(ID, ID+n);
  thrust::device_vector<int> idx(n);
  thrust::sequence(idx.begin(), idx.end(), 0);
  thrust::stable_sort(idx.begin(), idx.end(), 
		      compare<float>(thrust::raw_pointer_cast(&d_value[0])));
  
  //print(idx, "idx");
  
  for (int i=0; i<k; i++) {
    int j = idx[i];
    kval[i] = d_value[j];
    kID[i] = d_ID[j];
  }
}

void merge_neighbor_gpu(const float *D2PtrL, const int *IDl, int kl,
		        const float *D2PtrR, const int *IDr, int kr,
			float *nborDistPtr, int *nborIDPtr, int k) {
  thrust::device_vector<float> D2(kl+kr);
  thrust::device_vector<int> ID(kl+kr);
  thrust::copy(D2PtrL, D2PtrL+kl, D2.begin());
  thrust::copy(D2PtrR, D2PtrR+kr, D2.begin()+kl);
  thrust::copy(IDl, IDl+kl, ID.begin());
  thrust::copy(IDr, IDr+kr, ID.begin()+kl);
  
  //print(D2, "D2");
  //print(ID, "ID");

  // (sort and) unique
  thrust::device_vector<int> idx(kl+kr);
  thrust::sequence(idx.begin(), idx.end(), 0);
  thrust::stable_sort(idx.begin(), idx.end(), compare<int>(thrust::raw_pointer_cast(&ID[0])));
  thrust::stable_sort(ID.begin(), ID.end());
  
  //print(idx, "idx");
  //print(ID, "ID");
  
  thrust::device_vector<int> idx2(kl+kr);
  thrust::sequence(idx2.begin(), idx2.end(), 0);
  thrust::unique(idx2.begin(), idx2.end(), equal<int>(thrust::raw_pointer_cast(&ID[0])));
  ID.erase(thrust::unique(ID.begin(), ID.end()), ID.end());
  
  //print(idx2, "idx2");
  //print(ID, "ID");
  
  thrust::device_vector<float> d_value(ID.size());
  for (int i=0; i<ID.size(); i++) {
    int j = idx2[i];
    d_value[i] = D2[ idx[j] ];
  }
  
  //print(d_value, "d_value");
  
  // call k-select
  kselect_gpu(thrust::raw_pointer_cast(&d_value[0]), thrust::raw_pointer_cast(&ID[0]), ID.size(),
	      nborDistPtr, nborIDPtr, k);
}

