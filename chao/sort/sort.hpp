
#include "timer.hpp"

#include <thrust/random.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>

template <typename T>
using dvec = thrust::device_vector<T>;

void sort_matrix_rows_thrust(dvec<float> &A, dvec<int> &idx, int m, int n, double &t_sort);
void sort_matrix_rows_cub(dvec<float> &A, dvec<int> &idx, int m, int n, double &t_sort);
void sort_matrix_rows_cub2(dvec<float> &A, dvec<int> &idx, int m, int n, double &t_sort);
void sort_matrix_rows_mgpu(dvec<float>&, dvec<int>&, int, int, double &t_sort);

int* bitonic_mergesort(float *A, int m, int n);

