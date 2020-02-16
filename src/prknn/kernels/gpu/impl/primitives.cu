#include "primitives.hpp"

/*The CUDA Kernel*/
__global__ void vector_add_kernel(float *out, float *a, float *b, size_t n){
    for(size_t i = 0; i < n; ++i){
        out[i] = a[i] + b[i];
    }
}

/*Impl of function to be wrapped by Cython*/
/*Assume given data is on device*/
void vector_add(float *out, float *a, float *b, size_t n){
    vector_add_kernel<<<1, 1>>>(out, a, b, n);
}
    
