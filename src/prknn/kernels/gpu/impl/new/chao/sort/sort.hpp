#ifndef SORT_HPP
#define SORT_HPP

#include "mgpu_handle.hpp"

#include <thrust/device_vector.h>
#include <moderngpu/kernel_segsort.hxx>

template <typename T>
using dvec = thrust::device_vector<T>;
  
typedef dvec<int> ivec;
typedef dvec<float> fvec;


template <typename T>
void sort_matrix_rows_mgpu(dvec<T> &A, dvec<int> &idx, int N, dvec<int> &segments, int m) {
  auto& handle = mgpuHandle_t::instance();
  T *keys = thrust::raw_pointer_cast(A.data());
  int *vals = thrust::raw_pointer_cast(idx.data());
  int *segs = thrust::raw_pointer_cast(segments.data());
  mgpu::segmented_sort_indices(keys, vals, N, segs, m, mgpu::less_t<T>(), handle.mgpu_ctx());
}

template <typename T>
void sort_matrix_rows_mgpu(dvec<T> &A, dvec<int> &idx, int m, int n) { 
  dvec<int> segments(m); // m segments in all
  thrust::sequence(segments.begin(), segments.end(), 0, n);
  sort_matrix_rows_mgpu(A, idx, m*n, segments, m);
}


#endif
