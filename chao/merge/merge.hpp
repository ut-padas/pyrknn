#ifndef MERGE_GPU_HPP
#define MERGE_GPU_HPP

#include "util_gpu.hpp"


// Two merge routines for GPU:
//  - The first is the basic/intuitive one and was used in an early version of the randomized KNN code.
//  - The second is the optimized one and is now used in the randomized KNN code.

void merge_neighbors(float *nborD1, int *nborI1, const float *nborD2, const int *nborI2,
    int m, int n, int k);

// -----
// INPUT
// -----
// nborD1 & nborD2: two distance matrices to be merged, m-by-n matrices in row major 
// nborI1 & nborI2: IDs associated with the distance matrices, m-by-n matrices in row major
// m: number of rows
// n: number of columns
// k: number of nearest neighbors to compute

// ------
// OUTPUT
// ------
// nborD1: the first k-columns contain the k-smallest distances
// nborI1: the first k-columns contain the IDs associated with the k-smallest distances



void merge_neighbors(dvec<float> &nborDist, dvec<int> &nborID, int m, int n, int k, 
    float& t_sort, float& t_copy, float& t_unique);

// -----
// INPUT
// -----
// nborDist: distance matrix, m-by-n matrix in row major 
// nborID: IDs associated with the distance matrix, m-by-n matrix in row major
// m: number of rows
// n: number of columns
// k: number of nearest neighbors to compute

// ------
// OUTPUT
// ------
// nborDist: the first k-columns contain the k-smallest distances
// nborID: the first k-columns contain the IDs associated with the k-smallest distances
// t_sort, t_copy & t_unique: timings


#endif
