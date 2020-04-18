#include "util_gpu.hpp"
#include "mgpu_handle.hpp"
#include "timer_gpu.hpp"
#include "reorder.hpp"
#include "print.hpp"


void build_tree(dvec<float> &P, int n, int L, dvec<int> &order, int nBases, float &t_sort) {
  // initial order
  thrust::sequence(order.begin(), order.end(), 0);
  // permutation at every level (no need to initialize)
  ivec perm(n);
  int *idx = thrust::raw_pointer_cast(perm.data());
  auto& handle = mgpuHandle_t::instance();
  TimerGPU t;
  for (int i=0; i<L; i++) {
    int nNode = 1<<i;
    assert(n%nNode == 0);
    ivec seg(nNode);
    thrust::sequence(seg.begin(), seg.end(), 0, n/nNode);
    int *seghead = thrust::raw_pointer_cast(seg.data());
    float *val = thrust::raw_pointer_cast(P.data()+(i%nBases)*n);
    if (i>0) gather(val, n, order); // apply permutation of all previous levels
    //dprint(n, val, "value");
    t.start();
    mgpu::segmented_sort_indices(val, idx, n, seghead, nNode, mgpu::less_t<float>(), handle.mgpu_ctx()); 
    t.stop(); t_sort += t.elapsed_time();
    gather(order, perm);
    //dprint(n, val, "value sorted");
    //tprint(perm, "permutation");
    //tprint(order, "order");
  }
}


