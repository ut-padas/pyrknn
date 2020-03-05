#include "sort.hpp"
#include <moderngpu/kernel_segsort.hxx>

void sort_matrix_rows_mgpu(dvec<float> &A, dvec<int> &idx, int m, int n, double &t_sort) { 

  dvec<int> segments(m); // m segments in all
  for (int i=0; i<m; i++) segments[i] = i*n;
  float *keys = thrust::raw_pointer_cast(A.data());
  int *vals = thrust::raw_pointer_cast(idx.data());
  int *segs = thrust::raw_pointer_cast(segments.data());

  Timer t;
  mgpu::standard_context_t context(false);
  cudaDeviceSynchronize(); t.start();
  mgpu::segmented_sort_indices(keys, vals, m*n, segs, m, mgpu::less_t<float>(), context);
  cudaDeviceSynchronize(); t.stop();
  t_sort += t.elapsed_time();
}

