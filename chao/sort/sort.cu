#include <iostream>
#include <vector>

#include "timer.hpp"

#include <thrust/random.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>

struct prg : public thrust::unary_function<unsigned int, float>
{
    float a, b;

    __host__ __device__
    prg(float _a=0.f, float _b=1.f) : a(_a), b(_b) {};

    __host__ __device__
        float operator()(const unsigned int n) const
        {
            thrust::default_random_engine rng;
            thrust::uniform_real_distribution<float> dist(a, b);
            rng.discard(n);

            return dist(rng);
        }
};

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
using dvec = thrust::device_vector<T>;

int main(int argc, char *argv[]) {

  int m = 3;
  int n = 5;
  for (int i=1; i<argc; i++) {
    if (!strcmp(argv[i],"-n"))
      n = atoi(argv[i+1]);
    if (!strcmp(argv[i],"-m"))
      m = atoi(argv[i+1]);
  }
  assert(m > 0);
  assert(n > 0);

  dvec<float> A(m*n);

  thrust::counting_iterator<unsigned int> index(100);
  thrust::transform(index, index + m*n, A.begin(), prg());
  //print(A, m, n, "Initial");

  Timer t;
  cudaDeviceSynchronize(); t.start();
  for (int i=0; i<m; i++) {
    thrust::stable_sort(A.begin()+i*n, A.begin()+(i+1)*n);
  }
  cudaDeviceSynchronize(); t.stop();
  std::cout<<"Time for sequential sort: "<<t.elapsed_time()<<" s\n";
  //print(A, m, n, "Sequential sort");

  // batch call
  dvec<int> seg(m*n);
  for (int i=0; i<m; i++) {
    thrust::fill_n(seg.begin()+i*n, n, i);
  }
  //print(seg, m, n, "Initial value");
  
  cudaDeviceSynchronize(); t.start();
  thrust::stable_sort_by_key(A.begin(), A.end(), seg.begin());
  thrust::stable_sort_by_key(seg.begin(), seg.end(), A.begin());
  cudaDeviceSynchronize(); t.stop();
  std::cout<<"Time for batched sort: "<<t.elapsed_time()<<" s\n";
  
  //print(A, m, n, "Batched sort");


  return 0;
}


