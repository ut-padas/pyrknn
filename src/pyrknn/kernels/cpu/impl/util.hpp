#ifndef util_hpp
#define util_hpp

#include <algorithm>
#include <numeric> // std::iota

#include "omp.h"

int current_time_nanoseconds();


// helper functions to substitue their std counterparts
namespace par {

template <typename T>
void hcompute_range(T n, T &from , T &to) {
  int tid = omp_get_thread_num();
  int nThread = omp_get_num_threads();
  T block = n / nThread;
  T large = n % nThread;
  from = tid<large ? tid*(block+1) : tid*block+large;
  to = tid<large ? from+block+1 : from+block;
}


template <typename T>
void hiota(T begin, T end, int val) {
#pragma parallel
  {
    unsigned m = end - begin;
    unsigned from, to;
    hcompute_range(m, from, to);
    std::iota(begin+from, begin+to, val+from);
  }
}


template <class UnaryOperation>
void htransform(const float *begin, const float *end, float *output, UnaryOperation op) {
#pragma parallel
  {
    unsigned m = end - begin;
    unsigned from, to;
    hcompute_range(m, from, to);
    std::transform(begin+from, begin+to, output+from, op);
  }
}


template <typename T>
void hfill(T begin, T end, int val) {
#pragma parallel
  {
    unsigned m = end - begin;
    unsigned from, to;
    hcompute_range(m, from, to);
    std::fill(begin+from, begin+to, val);
  }
}

void hcopy(unsigned n, float *src, float *dst) {
  cblas_scopy(n, src, 1, dst, 1);
}

}


#endif
