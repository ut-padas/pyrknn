#ifndef SORT_HPP
#define SORT_HPP

#include "mgpu_handle.hpp"
#include "cuda_profiler_api.h"

#include <thrust/device_vector.h>
#include <moderngpu/kernel_segsort.hxx>
#include "math.h"

template <typename T>
using dvec = thrust::device_vector<T>;
  
typedef dvec<int> ivec;
typedef dvec<float> fvec;

__forceinline__  __device__ int signValue(float a) {
    return ((!signbit(a)) << 1) - 1;
}

template<typename type_t>
struct sign_less_t : public std::binary_function<type_t, type_t, bool> {
  __device__ __host__ bool operator()(type_t a, type_t b) const {
    return  (a < b && !isinf(a)) || ( !isnan(a) && (isnan(b) || isinf(b) ) );
  }
};

template <typename T>
void sort_matrix_rows_mgpu(dvec<T> &A, dvec<int> &idx, int N, dvec<int> &segments, int m) {
  auto& handle = mgpuHandle_t::instance();
  T *keys = thrust::raw_pointer_cast(A.data());
  int *vals = thrust::raw_pointer_cast(idx.data());
  int *segs = thrust::raw_pointer_cast(segments.data());
  cudaProfilerStart();
  mgpu::segmented_sort_indices(keys, vals, N, segs, m, sign_less_t<T>(), handle.mgpu_ctx());
  cudaProfilerStop();
}

template <typename T>
void sort_matrix_rows_mgpu(dvec<T> &A, dvec<int> &idx, int m, int n) { 
  dvec<int> segments(m); // m segments in all
  thrust::sequence(segments.begin(), segments.end(), 0, n);
  sort_matrix_rows_mgpu(A, idx, m*n, segments, m);
}


#endif
