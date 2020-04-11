#include "print.hpp"


void dprint(int m, int n, int nnz, 
    thrust::device_vector<int> &rowPtr, 
    thrust::device_vector<int> &colIdx, 
    thrust::device_vector<float> &val,
    const std::string &name) {
  dprint(m, n, nnz, 
      thrust::raw_pointer_cast(rowPtr.data()),
      thrust::raw_pointer_cast(colIdx.data()),
      thrust::raw_pointer_cast(val.data()),
      name);
}

