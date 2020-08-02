#ifndef REORDER_HPP
#define REORDER_HPP

#include "util_gpu.hpp"
#include "op_gpu.hpp"

void gather(dvec<int>&, const dvec<int>&);

void gather(float*, int, const dvec<int>&);

void gather(fvec&, int, int, const ivec&);

void gather(dvec<int>&, dvec<int>&, dvec<float>&, int, int, int, dvec<int>&);

template <typename T>
void scatter(dvec<T>& x, int m, int n, const dvec<int> &perm) {
  auto copy = x;
  const int *p = thrust::raw_pointer_cast(perm.data());
  auto zero = thrust::make_counting_iterator<int>(0);
  auto iter = thrust::make_transform_iterator(zero, permMatRow(p, n, n));
  auto P = thrust::make_permutation_iterator(x.begin(), iter); 
  thrust::copy(copy.begin(), copy.end(), P);
}


template <typename T>
void scatter(T *x, int m, int LD, int n, const dvec<int> &permutation) {
  
  dptr<T> xptr(x);

  // make a copy
  dvec<T> copy(m*n);
  auto zero = thrust::make_counting_iterator<int>(0);
  auto iter = thrust::make_transform_iterator(zero, firstKCols(n,LD));
  auto perm = thrust::make_permutation_iterator(xptr, iter);
  thrust::copy(perm, perm+m*n, copy.begin());
   
  const int *p = thrust::raw_pointer_cast(permutation.data());
  auto i = thrust::make_transform_iterator(zero, permMatRow(p, n, LD));
  auto P = thrust::make_permutation_iterator(xptr, i);
  thrust::copy(copy.begin(), copy.end(), P);
}


#endif
