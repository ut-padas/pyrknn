#ifndef MERGE_GPU_HPP
#define MERGE_GPU_HPP

void merge_neighbors(float*, const float*, int*, const int*, int, int,
    float*, int*, int, float&, bool debug=false);

void merge_neighbors_gpu(float *nborD1, int *nborI1, const float *nborD2, const int *nborI2,
    int m, int n, int k);

#endif
