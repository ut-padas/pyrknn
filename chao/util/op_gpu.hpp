#ifndef OP_GPU_HPP
#define OP_GPU_HPP

#include <thrust/random.h>
#include <thrust/functional.h>

struct prg: public thrust::unary_function<unsigned int, float> {
  int seed;
  float a, b;

  __host__ __device__
  prg(int s, float _a=0.f, float _b=1.f) : seed(s), a(_a), b(_b) {};

  __host__ __device__
  float operator()(const unsigned int n) const {
    thrust::default_random_engine rng(seed);
    thrust::uniform_real_distribution<float> dist(a, b);
    rng.discard(n);
    return dist(rng);
  }
};

struct rowIdx : public thrust::unary_function<int, int> {
  int nCol;

  __host__ __device__
    rowIdx(int c): nCol(c)  {}

  __host__ __device__
    int operator()(int i) {
      return i/nCol;
    }
};

struct firstKCols : public thrust::unary_function<int, int> {
  int k, LD;

  __host__ __device__
    firstKCols(int k_, int LD_): k(k_), LD(LD_)  {}

  __host__ __device__
    int operator()(int i) {
      return i/k*LD+i%k;
    }
};

struct permMatRow: public thrust::unary_function<int, int> {

  const int n;
  const int LD;
  const int *perm;

  __host__ __device__
  permMatRow(const int *p, int n_, int ld_): perm(p), n(n_), LD(ld_) {}

  __host__ __device__
  int operator()(int i) {
    return perm[i/n]*LD + i%n;
  }
};

struct strideBlock : public thrust::unary_function<int, int> {
  int k, m, N, LD, offset;

  __host__ __device__
    strideBlock(int k_, int m_, int N_, int ld, int o_): 
      k(k_), m(m_), N(N_), LD(ld), offset(o_) {}

  __host__ __device__
    int operator()(int i) {
      return i/(m*k)*(N*LD) + (i%(m*k))/k*LD + i%k + offset;
    }
};

struct firstKVals: public thrust::unary_function<int, int> {
  int k;
  int m;
  int N; // # points in every leaf node
  const int *permIdx; // from 0 to m*iLD 

  __host__ __device__
    firstKVals(int k_, int m_, int N_, const int *p_): 
      k(k_), m(m_), N(N_), permIdx(p_)  {}

  __host__ __device__
    int operator()(int i) {
      // i/k*N+i%k is the linear index for permIdx;
      // taking mod(N) is column index or local ID index.
      // i/(m*k)*N is the node index
      return permIdx[i/k*N+i%k]%N + i/(m*k)*N;
    }
};
#endif
