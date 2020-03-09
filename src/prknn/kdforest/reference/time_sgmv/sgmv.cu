#include <iostream>
#include <stdio.h>
#include <cublasXt.h>
#include <curand.h>

#include "timer.hpp"

#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    exit(EXIT_FAILURE);}} while(0)
#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    exit(EXIT_FAILURE);}} while(0)

void createRandomMatrix(curandGenerator_t &gen, int n, int d, void **matrix_gpu)
{
    CUDA_CALL(cudaMalloc(matrix_gpu, n*d*sizeof(float)));
    CURAND_CALL(curandGenerateUniform(gen,(float *) *matrix_gpu,n*d));
    
}

int main()
{
    cublasHandle_t handle;
    cublasCreate(&handle);
    curandGenerator_t gen;
    /* Create pseudo-random number generator */
    CURAND_CALL(curandCreateGenerator(&gen,CURAND_RNG_PSEUDO_DEFAULT));

    int d = 100;
    int n;
    float *matrix, *matrix_gpu, *vector_gpu;
    float *vector = (float *) malloc(d*sizeof(float));
    createRandomMatrix(gen,d,1,(void **) &vector_gpu);
    CUDA_CALL(cublasSetVector(d,sizeof(float),vector,1,vector_gpu,1));

    Timer timer = Timer();
   
    float alpha = 1.;
    float beta = 1.;

    for (int i = 1; i < 10; i++)
    {
        n = i * 1e5;
        matrix = (float *) calloc(n*d, sizeof(float));
        createRandomMatrix(gen,n,d,(void **) &matrix_gpu);
        
        CUDA_CALL(cublasSetMatrix(n,d,sizeof(float),matrix,n,matrix_gpu,n));
        
        float *result = (float *) calloc(n,sizeof(float));
        float *result_gpu;
        CUDA_CALL(cudaMalloc((void**) &result_gpu, n*sizeof(float)));
        CUDA_CALL(cublasSetVector(n,sizeof(float),result,1,result_gpu,1));

        timer.start();
        CUDA_CALL(cublasSgemv(handle, CUBLAS_OP_N,
            n, d,
            &alpha,
            matrix_gpu, n,
            vector_gpu, 1,
            &beta,
            result_gpu, 1));
        cudaDeviceSynchronize();
        timer.stop();
        timer.show_elapsed_time();

        CUDA_CALL(cublasGetVector(n,sizeof(float),result_gpu,1,result,1));

        CUDA_CALL(cudaFree(matrix_gpu));
        CUDA_CALL(cudaFree(result_gpu));
        free(matrix);
        free(result);
    }
    CUDA_CALL(cublasDestroy(handle));
    CURAND_CALL(curandDestroyGenerator(gen));
    CUDA_CALL(cudaFree(vector_gpu));
    free(vector);
    return 0;
}