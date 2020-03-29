#include "transpose.hpp"

#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/gather.h>
#include <thrust/binary_search.h>

template <typename T>
using dvec = thrust::device_vector<T>;
  
template <typename T>
using dptr = thrust::device_ptr<T>; 


void dprint(const dvec<int> &x) {
  for (int i=0; i<x.size(); i++)
    std::cout<<x[i]<<" ";
  std::cout<<std::endl;
}


void indices_to_offsets(const dvec<int> &idx, dvec<int> &offsets) {
  thrust::counting_iterator<int> zero(0);
  thrust::lower_bound(idx.begin(), idx.end(), zero, zero+offsets.size(), offsets.begin());
}


struct notEmpty {
  __host__ __device__
    bool operator()(const thrust::tuple<int, int> &t) {
      int curRowStart = thrust::get<0>(t);
      int nextRowStart = thrust::get<1>(t);
      return curRowStart != nextRowStart;
    }
};


void offsets_to_indices(const dvec<int> &offsets, dvec<int> &idx) {
  thrust::fill(idx.begin(), idx.end(), 0);
  thrust::counting_iterator<int> zero(0);
  auto hasRow = thrust::make_transform_iterator(
      thrust::make_zip_iterator(thrust::make_tuple(offsets.begin(), offsets.begin()+1)), notEmpty());
  thrust::scatter_if(zero, zero+offsets.size()-1, offsets.begin(), hasRow, idx.begin());
  thrust::inclusive_scan(idx.begin(), idx.end(), idx.begin(), thrust::maximum<int>());
}


void offsets_to_indices(const dptr<int> &offsets, int size,  dvec<int> &idx) {
  thrust::fill(idx.begin(), idx.end(), 0);
  thrust::counting_iterator<int> zero(0);
  auto hasRow = thrust::make_transform_iterator(
      thrust::make_zip_iterator(thrust::make_tuple(offsets, offsets+1)), notEmpty());
  thrust::scatter_if(zero, zero+size, offsets, hasRow, idx.begin());
  thrust::inclusive_scan(idx.begin(), idx.end(), idx.begin(), thrust::maximum<int>());
}


// the algorithm is inspired by the implementation in CUSP
void transpose_gpu(const int m, const int n, const int nnz, 
    const dvec<int> &rowPtrA, const dvec<int> &colIdxA, const dvec<float> &valA,
    dvec<int> &rowPtrT, dvec<int> &colIdxT, dvec<float> &valT) {
  dvec<int> permutation(nnz);
  thrust::sequence(permutation.begin(), permutation.end(), 0);
  dvec<int> idx = colIdxA;
  thrust::sort_by_key(idx.begin(), idx.end(), permutation.begin());
  // permute original value array
  thrust::gather(permutation.begin(), permutation.end(), valA.begin(), valT.begin());
  // compute row pointer of the tanspose
  indices_to_offsets(idx, rowPtrT);
  // permute original row indices
  offsets_to_indices(rowPtrA, idx);// form row indices
  thrust::gather(permutation.begin(), permutation.end(), idx.begin(), colIdxT.begin());
}


void transpose_gpu(int m, int n, int nnz, int *rowPtrA, int *colIdxA, float *valA,
    dvec<int> &rowPtrT, dvec<int> &colIdxT, dvec<float> &valT) {
  std::cout<<"[transpose] buffer: "<<nnz/1.e9*4*2<<" GB"<<std::endl;
  dvec<int> permutation(nnz);
  thrust::sequence(permutation.begin(), permutation.end(), 0);
  dvec<int> idx(dptr<int>(colIdxA), dptr<int>(colIdxA)+nnz);
  thrust::sort_by_key(idx.begin(), idx.end(), permutation.begin());
  // permute original value array
  thrust::gather(permutation.begin(), permutation.end(), dptr<float>(valA), valT.begin());
  // compute row pointer of the tanspose
  indices_to_offsets(idx, rowPtrT);
  // permute original row indices
  offsets_to_indices(dptr<int>(rowPtrA), m, idx);// form row indices
  thrust::gather(permutation.begin(), permutation.end(), idx.begin(), colIdxT.begin());
}


void transpose(const int m, const int n, const int nnz, 
    const int *rowPtrA, const int *colIdxA, const float *valA,
    int *rowPtrT, int *colIdxT, float *valT) {
  // copy data to device
  dvec<int> dRowPtrA(rowPtrA, rowPtrA+m+1);
  dvec<int> dColIdxA(colIdxA, colIdxA+nnz);
  dvec<float> dValA(valA, valA+nnz);

  dvec<int> dRowPtrT(n+1);
  dvec<int> dColIdxT(nnz);
  dvec<float> dValT(nnz);

  /*
  transpose_gpu(m, n, nnz,
      thrust::raw_pointer_cast(dRowPtrA.data()),
      thrust::raw_pointer_cast(dColIdxA.data()),
      thrust::raw_pointer_cast(dValA.data()),
      thrust::raw_pointer_cast(dRowPtrT.data()),
      thrust::raw_pointer_cast(dColIdxT.data()),
      thrust::raw_pointer_cast(dValT.data()));
  */

  transpose_gpu(m, n, nnz, dRowPtrA, dColIdxA, dValA, dRowPtrT, dColIdxT, dValT);

  // copy data to host
  thrust::copy(dRowPtrT.begin(), dRowPtrT.end(), rowPtrT);
  thrust::copy(dColIdxT.begin(), dColIdxT.end(), colIdxT);
  thrust::copy(dValT.begin(), dValT.end(), valT);
}


