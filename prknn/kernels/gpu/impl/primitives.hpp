#ifndef PRIMITIVES_GPU_HPP
#define PRIMITIVES_GPU_HPP

#include<stdio.h>

/* The function to be wrapped in Cython */
void vector_add(float *out, float *a, float *b, size_t n);

#endif //PRIMITIVES_GPU_HPP
