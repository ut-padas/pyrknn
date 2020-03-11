#include "sort_gpu.hpp"
#include <moderngpu/kernel_segsort.hxx>

mgpu::standard_context_t* sortGPU::ctxMGPU = NULL;

void sortGPU::init_mgpu() {
  sortGPU::ctxMGPU = new mgpu::standard_context_t(false);
}

void sortGPU::final_mgpu() {
  if (ctxMGPU != NULL) {
    delete ctxMGPU;
    ctxMGPU = NULL;
  }
}


void sortGPU::sort_matrix_rows_mgpu(dvec<float> &A, dvec<int> &idx, int N, 
    dvec<int> &segments, int m) { 

  if (sortGPU::ctxMGPU == NULL) sortGPU::init_mgpu();
  assert(sortGPU::ctxMGPU != NULL);

  float *keys = thrust::raw_pointer_cast(A.data());
  int *vals = thrust::raw_pointer_cast(idx.data());
  int *segs = thrust::raw_pointer_cast(segments.data());
  mgpu::segmented_sort_indices(keys, vals, N, segs, m, mgpu::less_t<float>(), *(sortGPU::ctxMGPU));
}


