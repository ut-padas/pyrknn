#ifndef MERGE_GPU_HPP
#define MERGE_GPU_HPP

void merge_neighbors(float*, const float*, int*, const int*, int, int,
    float*, int*, int, float&, bool debug=false);


void merge_neighbors_gpu(float*, int*, const float*, const int*, int, int, int,
    float&, float&, float&, float&, float&, float&, float&, float&, bool);

#endif
