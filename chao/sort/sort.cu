#include <iostream>
#include <vector>

#include <thrust/random.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>

template <typename T>
using dvec = thrust::device_vector<T>;

#include "sort_gpu.hpp"
#include "../util/util.hpp"

struct prg : public thrust::unary_function<unsigned int, float> {
  float a, b;

  __host__ __device__
  prg(float _a=0.f, float _b=1.f) : a(_a), b(_b) {};

  __host__ __device__
  float operator()(const unsigned int n) const {
    thrust::default_random_engine rng;
    thrust::uniform_real_distribution<float> dist(a, b);
    rng.discard(n);
    return dist(rng);
  }
};

struct diff_l2 : public thrust::binary_function<int, int, int> {
  __host__ __device__
  int operator()(const int a, const int b) const {
    return (a-b)*(a-b);
  }
};

int error_l2(const dvec<int> &x, const dvec<int> &y) {
  auto e = x;
  thrust::transform(x.begin(), x.end(), y.begin(), e.begin(), diff_l2());
  //print(e, "error");
  return thrust::reduce(e.begin(), e.end());
}

void random_initial(dvec<float> &A) {
  thrust::counting_iterator<unsigned int> index(100);
  thrust::transform(index, index + A.size(), A.begin(), prg());
}

void initialize(dvec<float> &A, dvec<int> &idx) {
  random_initial(A);
  thrust::sequence(idx.begin(), idx.end(), 0);
}

struct firstKCols : public thrust::unary_function<int, int> {
  int k, N;

  __host__ __device__
    firstKCols(int k_, int N_): k(k_), N(N_)  {}

  __host__ __device__
    int operator()(int i) {
      return i/k*N+i%k;
    }
};

template <typename T>
void get_kcols(const dvec<T> &D, dvec<T> &K, int m, int n, int k) {
  auto iter = thrust::make_transform_iterator(thrust::counting_iterator<int>(0), firstKCols(k, n));
  auto perm = thrust::make_permutation_iterator(D.begin(), iter);
  thrust::copy(perm, perm+m*k, K.begin());  
}

int main(int argc, char *argv[]) {

  int m = 3;
  int n = 5;
  int k = 3;
  int repeat = 10;
  bool debug = false;
  bool benchmark = false;
  for (int i=1; i<argc; i++) {
    if (!strcmp(argv[i],"-n"))
      n = atoi(argv[i+1]);
    if (!strcmp(argv[i],"-m"))
      m = atoi(argv[i+1]);
    if (!strcmp(argv[i],"-k"))
      k = atoi(argv[i+1]);
    if (!strcmp(argv[i],"-D"))
      debug = true;
    if (!strcmp(argv[i],"-B"))
      benchmark = true;
  }
  assert(m > 0);
  assert(n > 0);
  assert(k > 0 && k <= n);
  std::cout.precision(2);
  std::cout<<"\n======================\n"
           <<"Inputs:\n"
           <<"----------------------\n"
           <<"m: "<<m<<std::endl
           <<"n: "<<n<<std::endl
           <<"k: "<<k<<std::endl
           <<"----------------------\n"
           <<"repeat: "<<repeat<<std::endl
           <<"debug: "<<debug<<std::endl
           <<"benchmark: "<<benchmark<<std::endl
           <<"----------------------\n"
           <<std::scientific
           <<"Mem (input): "<<4.*m*n*2/1.e9<<" GB"<<std::endl
           <<"Mem (output): "<<4.*m*k*2/1.e9<<" GB"<<std::endl
           <<"======================\n\n";

  dvec<float> A(m*n);
  dvec<int> idx(m*n); // ID
  if (debug) {
    initialize(A, idx);
    print(A, m, n, "Initial matrix");
    print(idx, m, n, "Initial ID");
  }
  TimerGPU t;

  
  // MGPU
  double t_mgpu = 0.;
  sortGPU::init_mgpu();
  initialize(A, idx);
  t.start();
  sortGPU::sort_matrix_rows_mgpu(A, idx, m, n);
  t.stop(); t_mgpu += t.elapsed_time();

  if (benchmark) {
    t_mgpu = 0.;
    for (int i=0; i<repeat; i++) {
      initialize(A, idx);
      t.start();
      sortGPU::sort_matrix_rows_mgpu(A, idx, m, n); 
      t.stop(); t_mgpu += t.elapsed_time();
    }
    t_mgpu /= repeat;
  }
  sortGPU::final_mgpu();
 
  dvec<int> nbor_idx_mgpu(m*k);
  dvec<float> nbor_dist_mgpu(m*k);
  get_kcols(idx, nbor_idx_mgpu, m, n, k);
  get_kcols(A, nbor_dist_mgpu, m, n, k);
  
  if (debug) {
    print(nbor_idx_mgpu, m, k, "Modern GPU");
    print(nbor_dist_mgpu, m, k, "Modern GPU");
  }
  std::cout<<"Time for MGPU sort: "<<t_mgpu<<" s\n";


  // CUB
  double t_cub = 0.;
  sortGPU::init_cub(A, idx, m, n); 
  initialize(A, idx);
  t.start();
  sortGPU::sort_matrix_rows_cub(A, idx, m, n); 
  t.stop(); t_cub += t.elapsed_time();

  if (benchmark) {
    t_cub = 0.;
    for (int i=0; i<repeat; i++) {
      initialize(A, idx);
      t.start();
      sortGPU::sort_matrix_rows_cub(A, idx, m, n); 
      t.stop(); t_cub += t.elapsed_time();
    }
    t_cub /= repeat;
  }
  sortGPU::final_cub();

  dvec<int> nbor_idx_cub(m*k);
  dvec<float> nbor_dist_cub(m*k);
  get_kcols(idx, nbor_idx_cub, m, n, k);
  get_kcols(A, nbor_dist_cub, m, n, k);
  
  if (debug) {
    print(nbor_idx_cub, m, k, "CUB");
    print(nbor_dist_cub, m, k, "CUB");
  }
  std::cout<<"Time for CUB sort: "<<t_cub<<" s\n";


  // CUB2
  double t_cub2 = 0.;
  sortGPU::init_cub2(A, idx, m, n); 
  initialize(A, idx);
  t.start();
  sortGPU::sort_matrix_rows_cub2(A, idx, m, n); 
  t.stop(); t_cub2 += t.elapsed_time();

  if (benchmark) {
    t_cub2 = 0.;
    for (int i=0; i<repeat; i++) {
      initialize(A, idx);
      t.start();
      sortGPU::sort_matrix_rows_cub2(A, idx, m, n); 
      t.stop(); t_cub2 += t.elapsed_time();
    }
    t_cub2 /= repeat;
  }
  sortGPU::final_cub2();

  dvec<int> nbor_idx_cub2(m*k);
  dvec<float> nbor_dist_cub2(m*k);
  get_kcols(idx, nbor_idx_cub2, m, n, k);
  get_kcols(A, nbor_dist_cub2, m, n, k);
  
  if (debug) {
    print(nbor_idx_cub2, m, k, "CUB2");
    print(nbor_dist_cub2, m, k, "CUB2");
  }
  std::cout<<"Time for CUB2 sort: "<<t_cub2<<" s\n";


  // thrust 
  double t_thrust = 0.;
  initialize(A, idx);
  t.start();
  sortGPU::sort_matrix_rows_thrust(A, idx, m, n);
  t.stop(); t_thrust += t.elapsed_time();
  
  if (benchmark) {
    t_thrust = 0.;
    for (int i=0; i<repeat; i++) {
      initialize(A, idx);
      t.start();
      sortGPU::sort_matrix_rows_thrust(A, idx, m, n);
      t.stop(); t_thrust += t.elapsed_time();
    }
    t_thrust /= repeat;
  }
  
  dvec<int> nbor_idx_thrust(m*k);
  get_kcols(idx, nbor_idx_thrust, m, n, k);
  
  if (debug) {
    print(nbor_idx_thrust, m, k, "Thrust");
  }
  std::cout<<"Time for thrust sort: "<<t_thrust<<" s\n";
 

  // thrust2 
  double t_thrust2 = 0.;
  initialize(A, idx);
  t.start();
  sortGPU::sort_matrix_rows_thrust2(A, idx, m, n);
  t.stop(); t_thrust2 += t.elapsed_time();
  
  if (benchmark) {
    t_thrust2 = 0.;
    for (int i=0; i<repeat; i++) {
      initialize(A, idx);
      t.start();
      sortGPU::sort_matrix_rows_thrust2(A, idx, m, n);
      t.stop(); t_thrust2 += t.elapsed_time();
    }
    t_thrust2 /= repeat;
  }
  
  dvec<int> nbor_idx_thrust2(m*k);
  get_kcols(idx, nbor_idx_thrust2, m, n, k);
  
  if (debug) {
    print(nbor_idx_thrust2, m, k, "Thrust2");
  }
  std::cout<<"Time for thrust2 sort: "<<t_thrust2<<" s\n";
  
  /*
  // bitonic merge sort
  auto idx_bitonic = idx;
  {
    auto Acpy = A;
    float *dist = thrust::raw_pointer_cast(Acpy.data());
    int *idx = bitonic_mergesort(dist, m, n);

    cudaDeviceSynchronize();
    thrust::copy(idx, idx+m*n, idx_bitonic.begin());
  
    if (debug) print(Acpy, m, n, "Bitonic sort");
  }
  if (debug) print(idx_bitonic, "Bitonic sort");
  */


  // check results
  //std::cout<<"Error between MGPU and CUB: "<<error_l2(nbor_idx_mgpu, nbor_idx_cub)<<std::endl
    //       <<"Error between MGPU and Thrust: "<<error_l2(nbor_idx_mgpu, nbor_idx_thrust)<<std::endl
      //     <<"Error between MGPU and Thrust2: "<<error_l2(nbor_idx_mgpu, nbor_idx_thrust2)<<std::endl;
          

  return 0;
}


