#ifndef PRINT_GPU_HPP
#define PRINT_GPU_HPP

#include <thrust/device_vector.h>
#include <thrust/for_each.h>
#include <iostream>
#include <string>


void print(int m, int n, int nnz, int *rowPtr, int *colIdx, float *val, const std::string &);

void print(int m, int n, float *val, const std::string &);

void dprint(int m, int n, int nnz, int *rowPtr, int *colIdx, float *val, const std::string &);

template <typename T>
void print(T x, int n, std::string name) {
  std::cout<<name<<":\n";
  for (int i=0; i<n; i++)
    std::cout<<x[i]<<" ";
  std::cout<<std::endl;
}


void dprint(int, int*, const std::string&);
void dprint(int, float*, const std::string&);

void dprint(int, int, int*, const std::string&);
void dprint(int, int, float*, const std::string&);


template <typename T>
void print(const thrust::device_vector<T>& vec, const std::string &name) {
  std::cout<<std::endl<<name<<":"<<std::endl;
  for (int i=0; i<vec.size(); i++)
    std::cout<<vec[i]<<" ";
  std::cout<<std::endl<<std::endl;
}

template <typename T>
void print(const thrust::device_vector<T>& vec, int m, int n, const std::string &name) {
  std::cout<<std::endl<<name<<":"<<std::endl;
  for (int i=0; i<m; i++) {
    for (int j=0; j<n; j++)
      std::cout<<vec[i*n+j]<<" ";
    std::cout<<std::endl;
  }
}


template <typename T>
void tprint(const thrust::device_vector<T>& vec, const std::string &name) {
  std::cout<<name<<":"<<std::endl;
  for (int i=0; i<vec.size(); i++)
    std::cout<<vec[i]<<" ";
  std::cout<<std::endl<<std::endl;
}


template <typename T>
void tprint(int m, int n, const thrust::device_vector<T>& vec, const std::string &name) {
  std::cout<<name<<":"<<std::endl;
  for (int i=0; i<m; i++) {
    for (int j=0; j<n; j++)
      std::cout<<vec[i*n+j]<<" ";
    std::cout<<std::endl;
  }
  std::cout<<std::endl;
}

struct printf_functor
{
  __host__ __device__
  void operator()(int x)
  {
    // note that using printf in a __device__ function requires
    // code compiled for a GPU with compute capability 2.0 or
    // higher (nvcc --arch=sm_20)
    printf("%d ", x);
  }
};

template <typename T>
void iprint(T iter, int n, std::string name) {
  std::cout<<name<<std::endl;
  thrust::for_each(thrust::device, iter, iter+n, printf_functor());
  std::cout<<std::endl;
}

void dprint(int m, int n, int nnz, 
    thrust::device_vector<int> &rowPtr, 
    thrust::device_vector<int> &colIdx, 
    thrust::device_vector<float> &val,
    const std::string &name);

#endif
