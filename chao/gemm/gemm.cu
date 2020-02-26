#include <iostream>
#include <vector>

#include <thrust/random.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>

#include "cublas_v2.h"
#include <cuda_runtime.h>

#include "timer.hpp"

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

struct prg : public thrust::unary_function<unsigned int, float>
{
    float a, b;

    __host__ __device__
    prg(float _a=0.f, float _b=1.f) : a(_a), b(_b) {};

    __host__ __device__
        float operator()(const unsigned int n) const
        {
            thrust::default_random_engine rng;
            thrust::uniform_real_distribution<float> dist(a, b);
            rng.discard(n);

            return dist(rng);
        }
};

template <typename T>
using dvec = thrust::device_vector<T>;

int main(int argc, char *argv[]) {

  int S = 2;
  int N = 1024;
  int d = 64;

  // parse command line
  for (int i=1; i<argc; i++) {
    if (!strcmp(argv[i],"-s"))
      S = atoi(argv[i+1]);
    if (!strcmp(argv[i],"-n"))
      N = atoi(argv[i+1]);
    if (!strcmp(argv[i],"-d"))
      d = atoi(argv[i+1]);
  }
  assert(S>0);
  assert(N>0);
  assert(d>0);

  std::vector<dvec<float>> vecA(S), vecB(S), vecC(S);
  cudaStream_t str[S];
  // initialization  
  thrust::counting_iterator<unsigned int> index(0);
  for (int i=0; i<S; i++) {
    vecA[i].resize(N*d);
    vecB[i].resize(d*N);
    vecC[i].resize(N*N);
    thrust::transform(index, index + N*d, vecA[i].begin(), prg());
    thrust::transform(index, index + N*d, vecB[i].begin(), prg());
    cudaCheck( cudaStreamCreate(&str[i]) );
  }

  cublasHandle_t handle;
  cublasCheck( cublasCreate(&handle) );
  const float alpha = -2;
  const float beta = 0;

  // warming up
  cublasCheck( cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, N, d, &alpha,
	      thrust::raw_pointer_cast(vecA[0].data()), d,
	      thrust::raw_pointer_cast(vecB[0].data()), d, &beta,
	      thrust::raw_pointer_cast(vecC[0].data()), N) );

  // baseline
  const int repeat = 10;
  cudaDeviceSynchronize();
  Timer t; t.start();
  for (int r=0; r<repeat; r++) {
    for (int i=0; i<S; i++) {
      cublasCheck( cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, N, d, &alpha,
	      thrust::raw_pointer_cast(vecA[i].data()), d,
	      thrust::raw_pointer_cast(vecB[i].data()), d, &beta,
	      thrust::raw_pointer_cast(vecC[i].data()), N) );
    }
  }

  cudaDeviceSynchronize();
  t.stop();
  std::cout<<"GEMM baseline: "<<t.elapsed_time()/repeat<<" s"<<std::endl;

  // stream
  cudaDeviceSynchronize();
  t.start(); 
  for (int r=0; r<repeat; r++) {
  for (int i=0; i<S; i++) {
    cublasCheck( cublasSetStream(handle, str[i]) );
    cublasCheck( cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, N, d, &alpha,
	      thrust::raw_pointer_cast(vecA[i].data()), d,
	      thrust::raw_pointer_cast(vecB[i].data()), d, &beta,
	      thrust::raw_pointer_cast(vecC[i].data()), N) );
  }
  }

  cudaDeviceSynchronize();
  t.stop();
  std::cout<<"GEMM stream: "<<t.elapsed_time()/repeat<<" s"<<std::endl;

  // batched
  float *ptrA[S], *ptrB[S], *ptrC[S];
  for (int i=0; i<S; i++) {
    ptrA[i] = thrust::raw_pointer_cast(vecA[i].data());
    ptrB[i] = thrust::raw_pointer_cast(vecB[i].data());
    ptrC[i] = thrust::raw_pointer_cast(vecC[i].data());
  }
  
  cudaDeviceSynchronize();
  t.start();
  for(int r=0; r<repeat; r++) {
    cublasCheck( cublasSgemmBatched(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, N, d, &alpha,
			    ptrA, d, ptrB, d, &beta, ptrC, N, S) );
  }
  cudaDeviceSynchronize();
  t.stop();
  std::cout<<"GEMM batched: "<<t.elapsed_time()/repeat<<" s"<<std::endl;


  return 0;
}

