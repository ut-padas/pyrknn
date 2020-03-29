#ifndef SPKNN_HPP
#define SPKNN_HPP


void spknn(int*, int*, float*, int, int, int, int, int*, float*, int, int m=64);

void gemv_gpu(const int*, const int*, const float*, int, int, int,
    const float*, float*);

void batchedGemv_gpu(int *[], int *[], float *[],
    int*, int, int*, float *[], float *[], int);

void compute_distance_gpu(int *[], int *[], float *[], 
    int*, int, int*, float *, int, int, bool);

#endif
