#include "sort_gpu.hpp"
#include <thrust/functional.h>

namespace old {

void sortGPU::sort_matrix_rows_thrust(dvec<float> &A, dvec<int> &idx, int m, int n) {

  auto Acpy = A; 
  dvec<int> seg(m*n);
  auto itr = thrust::counting_iterator<int>(0);
  thrust::transform(itr, itr+m*n, seg.begin(), rowIdx(n));

  thrust::stable_sort_by_key(Acpy.begin(), Acpy.end(), seg.begin());
  thrust::stable_sort_by_key(A.begin(), A.end(), idx.begin());
  thrust::stable_sort_by_key(seg.begin(), seg.end(), idx.begin());
}


void sortGPU::sort_matrix_rows_thrust2(dvec<float> &A, dvec<int> &idx, int m, int n) {

  dvec<int> seg(m*n);
  auto itr = thrust::counting_iterator<int>(0);
  thrust::transform(itr, itr+m*n, seg.begin(), rowIdx(n));

  typedef thrust::device_vector<int>::iterator   IntIterator;
  typedef thrust::tuple<IntIterator, IntIterator> IteratorTuple;
  typedef thrust::zip_iterator<IteratorTuple> ZipIterator;
  ZipIterator zipItr(thrust::make_tuple(seg.begin(), idx.begin()));

  thrust::stable_sort_by_key(A.begin(), A.end(), zipItr);
  thrust::stable_sort_by_key(seg.begin(), seg.end(), idx.begin());
}

}
