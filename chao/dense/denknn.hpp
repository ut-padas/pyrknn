#ifndef DENKNN_HPP
#define DENKNN_HPP


void denknn(const int* hID, const float *hP, int n, int d, int level, int nTree,
        int *hNborID, float *hNborDist, int k, int blkPoint, int device);

// -----
// INPUT
// -----
// hID: ID array of input data points (on CPU)
// hP: coordinates of input data points (on CPU); n-by-d matrix stored in row major
// n: number of data points (length of hID array)
// d: dimension/number of coordinates of a data point

// ---------
// PARAMETER
// ---------
// level: number of tree level
// nTree: number of iterations for the randomized-KNN algorithm
// k: number of nearest neighbors to compute
// blkPoint: number of points in a batch for the leaf kernel
// device: GPU ID

// ------
// OUTPUT
// ------
// hNborID: IDs of KNN for all data points (on CUP); n-by-k matrix stored in row major
// hNborDist: distances of KNN for all data points (on CPU); n-by-k matrix in row major


#endif
