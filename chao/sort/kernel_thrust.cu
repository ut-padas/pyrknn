#include "sort.hpp"

void sort_matrix_rows_thrust(dvec<float> &A, dvec<int> &idx, int m, int n, double &t_sort) {
  dvec<int> seg(m*n);
  for (int i=0; i<m; i++) {
    thrust::fill_n(seg.begin()+i*n, n, i);
  }
  auto Acpy = A;
 
  Timer t;
  cudaDeviceSynchronize(); t.start();
  thrust::stable_sort_by_key(Acpy.begin(), Acpy.end(), seg.begin());
  thrust::stable_sort_by_key(A.begin(), A.end(), idx.begin());
  thrust::stable_sort_by_key(seg.begin(), seg.end(), idx.begin());
  cudaDeviceSynchronize(); t.stop();
  t_sort += t.elapsed_time();
}


