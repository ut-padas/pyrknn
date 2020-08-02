#include "util_gpu.hpp"
#include <cusolverDn.h>

#define CHECK_SOLVER(func) {                \
  cusolverStatus_t stat = (func);           \
  assert(CUSOLVER_STATUS_SUCCESS == stat);  \
}


void orthogonal(fvec &A, int m, int n) {

  assert(n <= m);
  float *d_A = thrust::raw_pointer_cast(A.data());

  cusolverDnHandle_t cusolverH = NULL;
  CHECK_SOLVER( cusolverDnCreate(&cusolverH) );
 
  int lwork_geqrf = 0;
  CHECK_SOLVER( cusolverDnSgeqrf_bufferSize(
        cusolverH,
        m,
        n,
        d_A,
        m,
        &lwork_geqrf) );

  int lwork_orgqr = 0;
  float *d_tau = NULL;
  CHECK_CUDA( cudaMalloc((void**)&d_tau, sizeof(float)*n) );
  CHECK_SOLVER( cusolverDnSorgqr_bufferSize(
        cusolverH,
        m,
        n,
        n,
        d_A,
        m,
        d_tau,
        &lwork_orgqr) );

  int lwork = (lwork_geqrf > lwork_orgqr)? lwork_geqrf : lwork_orgqr;
  float *d_work = NULL;
  CHECK_CUDA( cudaMalloc((void**)&d_work, sizeof(float)*lwork) );

  int *devInfo = NULL;
  CHECK_CUDA( cudaMalloc((void**)&devInfo, sizeof(int)) );
  CHECK_SOLVER( cusolverDnSgeqrf(
        cusolverH,
        m,
        n,
        d_A,
        m,
        d_tau,
        d_work,
        lwork,
        devInfo) );

  // check if QR is successful or not
  int info_gpu = 0;
  CHECK_CUDA( cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost) );
  assert(0 == info_gpu);

  CHECK_SOLVER( cusolverDnSorgqr(
        cusolverH,
        m,
        n,
        n,
        d_A,
        m,
        d_tau,
        d_work,
        lwork,
        devInfo) );
  CHECK_CUDA( cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost) );
  assert(0 == info_gpu);

  // free resource
  if (d_tau  ) CHECK_CUDA( cudaFree(d_tau) );
  if (devInfo) CHECK_CUDA( cudaFree(devInfo) );
  if (d_work ) CHECK_CUDA( cudaFree(d_work) );

  if (cusolverH) CHECK_SOLVER( cusolverDnDestroy(cusolverH) );
}


void orthogonal_gpu(float *hA, int m, int n) {
  fvec dA(hA, hA+m*n);
  orthogonal(dA, m, n);
  thrust::copy_n(dA.begin(), m*n, hA);
}


