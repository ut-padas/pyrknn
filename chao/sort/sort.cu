#include "sort.hpp"


void sort_gpu(float *hA, int m, int n) {
  fvec dA(hA, hA+m*n);
  ivec idx(m*n);
  ivec segments(m);
  thrust::sequence(segments.begin(), segments.end(), 0, n);
  sort_matrix_rows_mgpu(dA, idx, m*n, segments, m);
  thrust::copy_n(dA.begin(), m*n, hA);
}


