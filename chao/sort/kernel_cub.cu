#include "sort.hpp"
#include <cub/cub.cuh>

// requires extra memory as much as the input: A and idx
void sort_matrix_rows_cub(dvec<float> &A, dvec<int> &idx, int m, int n, double &t_sort) {
  dvec<int> ofs(m+1);
  for (int i=0; i<m+1; i++) ofs[i] = i*n;

  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  auto d_keys_in  = thrust::raw_pointer_cast(A.data());
  auto d_keys_out = thrust::raw_pointer_cast(A.data());
  auto d_vals_in  = thrust::raw_pointer_cast(idx.data());
  auto d_vals_out = thrust::raw_pointer_cast(idx.data());
  auto ofs_begin  = thrust::raw_pointer_cast(ofs.data());
  auto ofs_end    = thrust::raw_pointer_cast(ofs.data()+1);

  // compute temporary storage
  CubDebugExit( cub::DeviceSegmentedRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
        d_keys_in, d_keys_out, d_vals_in, d_vals_out,
        m*n, m, ofs_begin, ofs_end) );

  // allocate temporary storage
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  //std::cout<<"CUB temporary storage: "<<temp_storage_bytes/1e6<<" MB.\n";

  // run
  Timer t;
  cudaDeviceSynchronize(); t.start();
  CubDebugExit( cub::DeviceSegmentedRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
        d_keys_in, d_keys_out, d_vals_in, d_vals_out, m*n, m, ofs_begin, ofs_end) );
  cudaDeviceSynchronize(); t.stop();
  t_sort += t.elapsed_time();
  cudaFree(d_temp_storage);
}  


void sort_matrix_rows_cub2(dvec<float> &A, dvec<int> &idx, int m, int n, double &t_sort) {
  dvec<float> S(m*n);
  dvec<int> Icpy(m*n);
  dvec<int> ofs(m+1);
  for (int i=0; i<m+1; i++) ofs[i] = i*n;

  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  auto d_keys_buf1 = thrust::raw_pointer_cast(A.data());
  auto d_keys_buf2 = thrust::raw_pointer_cast(S.data());
  auto d_vals_buf1 = thrust::raw_pointer_cast(idx.data());
  auto d_vals_buf2 = thrust::raw_pointer_cast(Icpy.data());
  auto ofs_begin   = thrust::raw_pointer_cast(ofs.data());
  auto ofs_end     = thrust::raw_pointer_cast(ofs.data()+1);

  // Create a set of DoubleBuffers to wrap pairs of device pointers
  cub::DoubleBuffer<float> d_keys(d_keys_buf1, d_keys_buf2);
  cub::DoubleBuffer<int> d_values(d_vals_buf1, d_vals_buf2);

  // compute temporary storage
  CubDebugExit( cub::DeviceSegmentedRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, 
        d_keys, d_values, m*n, m, ofs_begin, ofs_end) );
  
  // allocate temporary storage
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  //std::cout<<"CUB temporary storage: "<<temp_storage_bytes/1e6<<" MB.\n";

  // run
  Timer t;
  cudaDeviceSynchronize(); t.start();
  CubDebugExit( cub::DeviceSegmentedRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, 
        d_keys, d_values, m*n, m, ofs_begin, ofs_end) );
  cudaDeviceSynchronize(); t.stop();
  t_sort += t.elapsed_time();
  cudaFree(d_temp_storage);
}


