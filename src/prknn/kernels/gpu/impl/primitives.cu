#include "primitives.hpp"

#include "thrust/sort.h"

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

void thrust_sort(float* array, size_t n, long device){
    cudaSetDevice(device);
    thrust::sort(array, array+n, [] __device__ (float a, float b){
        return (a > b);
    });
}
    
