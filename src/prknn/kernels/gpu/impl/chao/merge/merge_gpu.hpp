#ifndef MERGE_GPU_HPP
#define MERGE_GPU_HPP

void merge_neighbors(float*, const float*, int*, const int*, int, int,
    float*, int*, int, float&, bool debug=false);

void merge_neighbors_gpu(float *nborD1, int *nborI1, const float *nborD2, const int *nborI2, int m, int n, int k
#ifdef PROD
);
#else
, float &t_copy, float &t_sort1, float &t_unique, float &t_copy2, float &t_seg, float &t_sort2, float &t_out, float &t_kernel, bool debug);
#endif

#endif //MERGE_GPU_HPP
