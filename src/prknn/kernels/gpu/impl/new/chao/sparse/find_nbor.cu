#include "util_gpu.hpp"
#include "timer_gpu.hpp"
#include "mgpu_handle.hpp"
#include "op_gpu.hpp"
#include "print.hpp"


void get_kcols_dist(const dvec<float> &D, float *Dk,
    int nLeaf, int m, int LD, int k, int N, int offset) {
  auto zero  = thrust::make_counting_iterator<int>(0);
  auto iterD = thrust::make_transform_iterator(zero, firstKCols(k, N));
  auto permD = thrust::make_permutation_iterator(D.begin(), iterD);
  auto iterK = thrust::make_transform_iterator(zero, strideBlock(k, m, N, LD, offset));
  auto permK = thrust::make_permutation_iterator(thrust::device_ptr<float>(Dk), iterK);
  thrust::copy(permD, permD+nLeaf*m*k, permK);
}


void get_kcols_ID(const dvec<int> &permIdx, int *IDk, const int *ID,
    int nLeaf, int m, int LD, int k, int N, int offset) {
  const int *pIdx  = thrust::raw_pointer_cast(permIdx.data());
  auto zero  = thrust::make_counting_iterator<int>(0);
  auto iterD = thrust::make_transform_iterator(zero, firstKVals(k, m, N, pIdx));
  auto permD = thrust::make_permutation_iterator(thrust::device_ptr<const int>(ID), iterD);
  auto iterK = thrust::make_transform_iterator(zero, strideBlock(k, m, N, LD, offset));
  auto permK = thrust::make_permutation_iterator(thrust::device_ptr<int>(IDk), iterK);
  thrust::copy(permD, permD+nLeaf*m*k, permK);
}


void find_neighbor(dvec<float> &Dist, int *ID, float *nborDist, int *nborID,
    int nLeaf, int m, int LD, int k, int N, int offset, 
    dvec<int> &idx, float &t_sort, float &t_kcol) {
  
  // sorting
  dvec<int> segments(m*nLeaf);
  thrust::sequence(segments.begin(), segments.end(), 0, N);

  float *keys = thrust::raw_pointer_cast(Dist.data());
  int *vals = thrust::raw_pointer_cast(idx.data());
  int *segs = thrust::raw_pointer_cast(segments.data());
 
  //dprint(m*nLeaf, N, keys, "distance");
  //float t_sort, t_copy;

  auto& handle = mgpuHandle_t::instance();
  TimerGPU t;
  t.start();
  mgpu::segmented_sort_indices(keys, vals, m*nLeaf*N, segs, m*nLeaf, mgpu::less_t<float>(), handle.mgpu_ctx());  
  t.stop(); t_sort += t.elapsed_time();
  
  //dprint(m*nLeaf, N, keys, "[find nbor] sorted distance");
  //dprint(m*nLeaf, N, vals, "[find nbor] index");
  

  // get first k
  t.start();
  get_kcols_dist(Dist, nborDist, nLeaf, m, LD, k, N, offset);
  get_kcols_ID(idx, nborID, ID, nLeaf, m, LD, k, N, offset);
  t.stop(); t_kcol += t.elapsed_time();
}


