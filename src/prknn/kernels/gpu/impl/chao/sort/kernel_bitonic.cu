#include <iostream>
#include <cuda_runtime.h>

#include "sort_gpu.hpp"

using namespace std;
namespace old {

inline void check(cudaError_t status, string error) {
  if (status != cudaSuccess) {
    cout << error << endl;
    exit(1);
  }
}

__global__
void iota_fill(int *indices, int m, int n) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;

  int i = index / n;
  int j = index % n;

  // Cuda store column major.
  if (i < m && j < n)
    indices[i + j * m] = j;
}

__global__
void bitonic_mergesort_step(float *C, int *indices, int split, int away, int m, int n) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (index >= n)
    return;
  else if (row >= m)
    return;

  int i_mask = ((1 << 30) - 1) - away;
  int j_mask = away;
  int is_inc_mask = split << 1;

  int i = index & i_mask;
  int j = index | j_mask;

  int is_inc = (index & is_inc_mask) == 0;

  if (index == j)
    return;

  bool need_swap = false;
  need_swap |= (is_inc && (C[row + i * m] > C[row + j * m]));
  need_swap |= (!is_inc && (C[row + i * m] < C[row + j * m]));

  if (need_swap) {
    float tmp_C = C[row + i * m];
    int tmp_indices = indices[row + i * m];

    C[row + i * m] = C[row + j * m];
    C[row + j * m] = tmp_C;

    indices[row + i * m] = indices[row + j * m];
    indices[row + j * m] = tmp_indices;
  }
}

int* bitonic_mergesort(float *C, int m, int n) {
  int *indices;

  check(cudaMalloc((void**) &indices, m * n * sizeof(int)),
        "initialize indices");

  iota_fill<<<(m * n + 255) / 256, 256>>>(indices, m, n);

  dim3 blocks(32, 32);
  dim3 grids;
  grids.x = (n + blocks.x - 1) / blocks.x;
  grids.y = (m + blocks.y - 1) / blocks.y;

  for (int split = 1; split < n; split <<= 1)
    for (int away = split; away >= 1; away >>= 1)
      bitonic_mergesort_step<<<grids, blocks>>>(C, indices, split, away, m, n);

  return indices;
}

/*
void k_select(T *distances_device, int *indices_device,
               vector<T> &distances, vector<int> &indices,
               int m, int n, int k) {
  distances.resize(m * k);
  indices.resize(m * k);

  check(cudaMemcpy(&distances[0], distances_device, (size_t) ((m * k) * sizeof(T)),
                   cudaMemcpyDeviceToHost),
        "copy device to host (distances)");

  check(cudaMemcpy(&indices[0], indices_device, (size_t) ((m * k) * sizeof(int)),
                   cudaMemcpyDeviceToHost),
        "copy device to host (indices)");
}
*/


}
