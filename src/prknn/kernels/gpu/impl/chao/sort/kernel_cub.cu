#include "sort_gpu.hpp"
#include <cub/cub.cuh>


size_t sortGPU::storage_bytes_cub = 0;
void*  sortGPU::storage_cub = NULL;


void sortGPU::init_cub(dvec<float> &A, dvec<int> &idx, int m, int n) {
  dvec<int> ofs(m+1);
  auto itr = thrust::counting_iterator<int>(0);
  thrust::transform(itr, itr+m+1, ofs.begin(), stride(n));

  auto d_keys_in  = thrust::raw_pointer_cast(A.data());
  auto d_keys_out = thrust::raw_pointer_cast(A.data());
  auto d_vals_in  = thrust::raw_pointer_cast(idx.data());
  auto d_vals_out = thrust::raw_pointer_cast(idx.data());
  auto ofs_begin  = thrust::raw_pointer_cast(ofs.data());
  auto ofs_end    = thrust::raw_pointer_cast(ofs.data()+1);

  // compute temporary storage
  CubDebugExit( cub::DeviceSegmentedRadixSort::SortPairs(
        sortGPU::storage_cub, sortGPU::storage_bytes_cub,
        d_keys_in, d_keys_out, d_vals_in, d_vals_out,
        m*n, m, ofs_begin, ofs_end) );

  // allocate temporary storage
  cudaMalloc(&storage_cub, storage_bytes_cub);
}  


void sortGPU::final_cub() {
  if (storage_cub != NULL) {
    cudaFree(storage_cub);
    storage_cub = NULL;
    storage_bytes_cub = 0;
  }
}


void sortGPU::sort_matrix_rows_cub(dvec<float> &A, dvec<int> &idx, int m, int n) {

  if (storage_cub == NULL) init_cub(A, idx, m, n);
  assert(storage_cub != NULL);

  dvec<int> ofs(m+1);
  auto itr = thrust::counting_iterator<int>(0);
  thrust::transform(itr, itr+m+1, ofs.begin(), stride(n));

  auto d_keys_in  = thrust::raw_pointer_cast(A.data());
  auto d_keys_out = thrust::raw_pointer_cast(A.data());
  auto d_vals_in  = thrust::raw_pointer_cast(idx.data());
  auto d_vals_out = thrust::raw_pointer_cast(idx.data());
  auto ofs_begin  = thrust::raw_pointer_cast(ofs.data());
  auto ofs_end    = thrust::raw_pointer_cast(ofs.data()+1);

  // run
  CubDebugExit( cub::DeviceSegmentedRadixSort::SortPairs(storage_cub, storage_bytes_cub,
        d_keys_in, d_keys_out, d_vals_in, d_vals_out, m*n, m, ofs_begin, ofs_end) );
}  


size_t sortGPU::storage_bytes_cub2 = 0;
void*  sortGPU::storage_cub2 = NULL;
sortGPU::dvec<float> sortGPU::Acpy;
sortGPU::dvec<int>   sortGPU::Icpy;

void sortGPU::init_cub2(dvec<float> &A, dvec<int> &idx, int m, int n) {
  sortGPU::Acpy = A;
  sortGPU::Icpy = idx;

  dvec<int> ofs(m+1);
  auto itr = thrust::counting_iterator<int>(0);
  thrust::transform(itr, itr+m+1, ofs.begin(), stride(n));

  auto d_keys_buf1 = thrust::raw_pointer_cast(A.data());
  auto d_keys_buf2 = thrust::raw_pointer_cast(Acpy.data());
  auto d_vals_buf1 = thrust::raw_pointer_cast(idx.data());
  auto d_vals_buf2 = thrust::raw_pointer_cast(Icpy.data());
  auto ofs_begin   = thrust::raw_pointer_cast(ofs.data());
  auto ofs_end     = thrust::raw_pointer_cast(ofs.data()+1);

  // Create a set of DoubleBuffers to wrap pairs of device pointers
  cub::DoubleBuffer<float> d_keys(d_keys_buf1, d_keys_buf2);
  cub::DoubleBuffer<int> d_values(d_vals_buf1, d_vals_buf2);

  // compute temporary storage
  CubDebugExit( cub::DeviceSegmentedRadixSort::SortPairs(storage_cub2, storage_bytes_cub2, 
        d_keys, d_values, m*n, m, ofs_begin, ofs_end) );
  
  // allocate temporary storage
  cudaMalloc(&storage_cub2, storage_bytes_cub2);
}


void sortGPU::final_cub2() {
  if (storage_cub2 != NULL) {
    cudaFree(storage_cub2);
    storage_cub2 = NULL;
    storage_bytes_cub2 = 0;
    Acpy.clear();
    Icpy.clear();
    Acpy.shrink_to_fit();
    Icpy.shrink_to_fit();
  }
}


void sortGPU::sort_matrix_rows_cub2(dvec<float> &A, dvec<int> &idx, int m, int n) {
  
  if (storage_cub2 == NULL) init_cub2(A, idx, m, n);
  assert(storage_cub2 != NULL);

  dvec<int> ofs(m+1);
  auto itr = thrust::counting_iterator<int>(0);
  thrust::transform(itr, itr+m+1, ofs.begin(), stride(n));

  auto d_keys_buf1 = thrust::raw_pointer_cast(A.data());
  auto d_keys_buf2 = thrust::raw_pointer_cast(sortGPU::Acpy.data());
  auto d_vals_buf1 = thrust::raw_pointer_cast(idx.data());
  auto d_vals_buf2 = thrust::raw_pointer_cast(sortGPU::Icpy.data());
  auto ofs_begin   = thrust::raw_pointer_cast(ofs.data());
  auto ofs_end     = thrust::raw_pointer_cast(ofs.data()+1);

  // Create a set of DoubleBuffers to wrap pairs of device pointers
  cub::DoubleBuffer<float> d_keys(d_keys_buf1, d_keys_buf2);
  cub::DoubleBuffer<int> d_values(d_vals_buf1, d_vals_buf2);

  // run
  CubDebugExit( cub::DeviceSegmentedRadixSort::SortPairs(storage_cub2, storage_bytes_cub2, 
        d_keys, d_values, m*n, m, ofs_begin, ofs_end) );
}


