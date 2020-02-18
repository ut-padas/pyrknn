#ifndef PRIMITIVES_GPU_HPP
#define PRIMITIVES_GPU_HPP

#include<stdio.h>

/* The function to be wrapped in Cython */
void vector_add(float *out, float *a, float *b, size_t n);

void device_reduce_warp_atomic(const float *in, float* out, size_t n);

float device_kelley_cutting(float *arr, const size_t n);
 
#endif //PRIMITIVES_GPU_HPP
