#ifndef util_hpp
#define util_hpp

#include <algorithm>
#include <numeric> // std::iota

#ifndef PYRKNN_USE_MKL
    #include <cblas.h>
#else
    #include<mkl.h>
#endif

#include "omp.h"

int current_time_nanoseconds();


// helper functions to substitue their std counterparts
namespace par {


    template <typename T>
    void compute_range(T n, T &from , T &to) {
      int tid = omp_get_thread_num();
      int nThread = omp_get_num_threads();
      T block = n / nThread;
      T large = n % nThread;
      from = tid<large ? tid*(block+1) : tid*block+large;
      to = tid<large ? from+block+1 : from+block;
    }


    template <typename T>
    void iota(T begin, T end, int val) {
    #pragma parallel
      {
        unsigned m = end - begin;
        unsigned from, to;
        compute_range(m, from, to);
        std::iota(begin+from, begin+to, val+from);
      }
    }


    template <class UnaryOperation>
    void transform(const float *begin, const float *end, float *output, UnaryOperation op) {
    #pragma parallel
      {
        unsigned m = end - begin;
        unsigned from, to;
        compute_range(m, from, to);
        std::transform(begin+from, begin+to, output+from, op);
      }
    }


    template <typename T>
    void fill(T begin, T end, int val) {
    #pragma parallel
      {
        unsigned m = end - begin;
        unsigned from, to;
        compute_range(m, from, to);
        std::fill(begin+from, begin+to, val);
      }
    }


}


#endif
