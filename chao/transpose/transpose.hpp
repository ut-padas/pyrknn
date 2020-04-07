#ifndef TRANSPOSE_GPU_HPP
#define TRANSPOSE_GPU_HPP

#include "util_gpu.hpp"

void transpose(const int, const int, const int, int*, int*, float*,
    ivec&, ivec&, fvec&);

void transpose(const int, const int, const int, const ivec&, const ivec&, const fvec&,
    ivec&, ivec&, fvec&);

#endif

