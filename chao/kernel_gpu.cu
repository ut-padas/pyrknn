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


void bb_gemm_kselect(int nLeaf, float *ptrR[], float *ptrQ[], int *ptrID[], int N, int d, 
		     float *ptrNborDist[], int *ptrNborID[], int k, bool debug) {
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

  double t_dist = 0., t_sort = 0., t_store = 0.;
  Timer t, t1;
  cudaDeviceSynchronize(); std::cout<<std::endl;
  t.start();

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
  for (int i=0; i<nLeaf; i++) {
    thrust::transform_iterator<square, FloatIterator> R_iter(vecR[i].begin(), square());
    thrust::transform_iterator<square, FloatIterator> Q_iter(vecQ[i].begin(), square());
    //thrust::reduce_by_key(thrust::cuda::par.on(str[i]), 
    thrust::reduce_by_key(
		    countItr, countItr+N*d, R_iter, vecRrow[i].begin(), vecR2[i].begin());
    //thrust::reduce_by_key(thrust::cuda::par.on(str[i]), 
    thrust::reduce_by_key(
		    countItr, countItr+N*d, Q_iter, vecQrow[i].begin(), vecQ2[i].begin());
    if (debug) {
      print(vecR2[i], "R2");
      print(vecQ2[i], "Q2");
    }
  }

  cudaDeviceSynchronize();
  t.stop();
  std::cout<<"Time for computing norm: "<<t.elapsed_time()<<" s"<<std::endl;

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
  for (int r=0; r<M; r++) {
    float *ptrdQ[nLeaf];
    for (int i=0; i<nLeaf; i++) {
      ptrdQ[i] = thrust::raw_pointer_cast(vecQ[i].data()+r*d*d);
    }

    cudaDeviceSynchronize();
    t1.start();
    // compute the distance (transpose) (cublas assumes column-major ordering)
    cublasCheck( cublasSgemmBatched(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, d, d, &alpha,
          ptrdR, d, ptrdQ, d, &beta, ptrDist, N, nLeaf) );
    // rank-1 updates
    for (int i=0; i<nLeaf; i++) {
      if (debug) {
        print(vecDist[i], "Dist (before rank-1 update)");
      }
      //cublasCheck( cublasSetStream(handle, streams[i]) );
      cublasCheck( cublasSger(handle, N, d, &one, 
			      thrust::raw_pointer_cast(vecR2[i].data()), 1, 
			      thrust::raw_pointer_cast(ones.data()), 1,
			      ptrDist[i], N) );
      cublasCheck( cublasSger(handle, N, d, &one, 
			      thrust::raw_pointer_cast(ones.data()), 1, 
			      thrust::raw_pointer_cast(vecQ2[i].data()+r*d), 1,
			      ptrDist[i], N) );
      if (debug) {
        print(vecDist[i], "Dist");
      }
    }
    cudaDeviceSynchronize();
    t1.stop(); t_dist += t1.elapsed_time();

    // k-select
    for (int i=0; i<nLeaf; i++) {
      dvec<int> idx(N*d);
      dvec<int> seg(N*d);
      for (int q=0; q<d; q++) {
        thrust::sequence(idx.begin()+q*N, idx.begin()+(q+1)*N, 0);  // initialize indices
        thrust::fill_n(seg.begin()+q*N, N, q);
      }
      
      auto vecDistCpy1 = vecDist[i];
      auto vecDistCpy2 = vecDist[i];
      
      cudaDeviceSynchronize();
      t1.start();
      
      thrust::stable_sort_by_key(vecDistCpy1.begin(), vecDistCpy1.end(), idx.begin());
      thrust::stable_sort_by_key(vecDistCpy2.begin(), vecDistCpy2.end(), seg.begin());

      // the following zipped version is slower than above
      //thrust::stable_sort_by_key(vecDistCpy1.begin(), vecDistCpy1.end(),
      //    thrust::make_zip_iterator( make_tuple(idx.begin(), seg.begin()) ));
      
      thrust::stable_sort_by_key(seg.begin(), seg.end(), idx.begin());

      cudaDeviceSynchronize();
      t1.stop(); t_sort += t1.elapsed_time();
      t1.start();

      dvec<int> idx2(d*k);
      thrust::sequence(idx2.begin(), idx2.end(), 0);
      auto iter = thrust::make_transform_iterator(idx2.begin(), module<int>(k, N, thrust::raw_pointer_cast(idx.data())));
      auto permID = thrust::make_permutation_iterator(vecID[i].begin(), iter);
      thrust::copy(permID, permID+d*k, vecNborID[i].begin()+r*d*k);

      auto iter2 = thrust::make_transform_iterator(idx2.begin(), module2(k, N, thrust::raw_pointer_cast(idx.data())));

      auto permID2 = thrust::make_permutation_iterator(vecDist[i].begin(), iter2);
      thrust::copy(permID2, permID2+d*k, vecNborDist[i].begin()+r*d*k);
      

      cudaDeviceSynchronize();
      t1.stop(); t_store += t1.elapsed_time();
    }
  }
  
  cudaDeviceSynchronize();
  t.stop();
  std::cout<<"Time for distance: "<<t_dist<<" s\n"
           <<"Time for sort: "<<t_sort<<" s\n"
           <<"Time for store: "<<t_store<<" s\n";
  std::cout<<"Time for GEMM-kselect: "<<t.elapsed_time()<<" s\n"<<std::endl;

  // copy results back to host
  for (int i=0; i<nLeaf; i++) {
    cudaStreamSynchronize(streams[i]);
    cudaStreamDestroy(streams[i]);
    if (debug) {
      print(vecNborDist[i], "Neighbor distance");
      print(vecNborID[i], "Neighbor ID");
    }
    thrust::copy(vecNborDist[i].begin(), vecNborDist[i].end(), ptrNborDist[i]);
    thrust::copy(vecNborID[i].begin(), vecNborID[i].end(), ptrNborID[i]);
  }
  cublasDestroy(handle);
}

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


