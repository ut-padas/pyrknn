#ifndef SPKNN_HPP
#define SPKNN_HPP


void spknn(int *hID, int *hRowPtr, int *hColIdx, float *hVal, 
    int n, int d, int nnz, int level, int nTree,
    int *hNborID, float *hNborDist, int k, 
    int blkLeaf, int blkPoint, int device);

// -----
// INPUT
// -----
// hID: ID array of input data points (on CPU)
// hRowPtr, hColIdx & hVal: coordinates of input data points (on CPU); n-by-d SARSE matrix stored in compress sparse row (CSR) format
// n: number of data points (length of hID array)
// d: dimension/number of coordinates of a data point
// nnz: number of nonzero coordinates

// ---------
// PARAMETER
// ---------
// level: number of tree level
// nTree: number of iterations for the randomized-KNN algorithm
// k: number of nearest neighbors to compute
// blkLeaf: number of leaf nodes processed at a time in the leaf kernel (used to reduce memory footprint)
// blkPoint: number of points in a batch for the leaf kernel
// device: GPU ID

// ------
// OUTPUT
// ------
// hNborID: IDs of KNN for all data points (on CUP); n-by-k matrix stored in row major
// hNborDist: distances of KNN for all data points (on CPU); n-by-k matrix in row major



#endif
