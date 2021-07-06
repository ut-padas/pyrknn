#ifndef MERGE_GPU_HPP
#define MERGE_GPU_HPP

#include "util_gpu.hpp"

void merge_neighbors(float *nborD1, int *nborI1, const float *nborD2, const int *nborI2,
    int m, int n, int k);

void merge_neighbors(dvec<float> &, dvec<int> &, int m, int n, int k, 
    float&, float&, float&);


#endif
