#include <thrust/random.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/random/linear_congruential_engine.h>
#include <thrust/random/normal_distribution.h>

struct prg : public thrust::unary_function<unsigned int, float> {
  float a, b;

  __host__ __device__
  prg(float _a=0.f, float _b=1.f) : a(_a), b(_b) {};

  __host__ __device__
  float operator()(const unsigned int n) const {
    thrust::minstd_rand rng(n);
    thrust::random::normal_distribution<float> dist(a, b);
    //thrust::default_random_engine rng;
    //thrust::uniform_real_distribution<float> dist(a, b);
    rng.discard(n);
    return dist(rng);
  }
};


void init_random_gpu(float *h_ptr, int N) {
  thrust::device_vector<float> A(N);
  thrust::counting_iterator<unsigned int> index(100);
  thrust::transform(index, index + N, A.begin(), prg());
  thrust::copy(A.begin(), A.end(), h_ptr);
}


