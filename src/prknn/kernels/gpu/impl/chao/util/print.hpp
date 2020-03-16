#ifndef PRINT_GPU_HPP
#define PRINT_GPU_HPP

#include <thrust/device_vector.h>
#include <iostream>

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

#endif
