#ifndef SORT_GPU_HPP
#define SORT_GPU_HPP

#include <thrust/device_vector.h>
#include <moderngpu/kernel_segsort.hxx>

class sortGPU {

public:

  template <typename T>
  using dvec = thrust::device_vector<T>;

  static void init_mgpu();
  static void final_mgpu();
  static void sort_matrix_rows_mgpu(dvec<float>&, dvec<int>&, int, int);

  static void init_cub(dvec<float> &A, dvec<int> &idx, int m, int n);
  static void final_cub();
  static void sort_matrix_rows_cub(dvec<float> &A, dvec<int> &idx, int m, int n);
  
  static void init_cub2(dvec<float> &A, dvec<int> &idx, int m, int n);
  static void final_cub2();
  static void sort_matrix_rows_cub2(dvec<float> &A, dvec<int> &idx, int m, int n);

  static void sort_matrix_rows_thrust(dvec<float> &A, dvec<int> &idx, int m, int n);
  static void sort_matrix_rows_thrust2(dvec<float> &A, dvec<int> &idx, int m, int n);

  //int* bitonic_mergesort(float *A, int m, int n);

  /*
  ~sortGPU() {
    final_mgpu();
    final_cub();
    final_cub2();
    final_thrust();
  }
  */

private:

  static mgpu::standard_context_t *ctxMGPU;

  static size_t storage_bytes_cub;
  static void *storage_cub;

  static size_t storage_bytes_cub2;
  static void *storage_cub2;
  static dvec<float> Acpy;
  static dvec<int> Icpy;

};

struct stride: public thrust::unary_function<int, int> {
  const int S;
  
  __host__ __device__
    stride(int _S): S(_S) {}

  __host__ __device__
    int operator()(const int i) const {
      return i*S;
    }
};

struct rowIdx: public thrust::unary_function<int, int> {
  const int nCols;
  
  __host__ __device__
    rowIdx(int n): nCols(n) {}

  __host__ __device__
    int operator()(const int i) const {
      return i/nCols;
    }
};


#endif
