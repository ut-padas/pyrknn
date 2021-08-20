#ifndef SPKNN_HPP
#define SPKNN_HPP


// this one seems not being used anywhere
void spknn(int*, int*, int*, float*, int, int, int, int*, float*, int, int, int, int, int);



void spknn
(unsigned *ID, int *rowPtr, int *colIdx, float *val, unsigned n, unsigned d, unsigned nnz, 
 unsigned *nborID, float *nborDist, int k, int level, int nTree, int blkPoint, int cores)

// -----
// INPUT
// -----
// ID: ID array of input data points
// rowPtr, colIdx & val: coordinates of input data points; n-by-d SARSE matrix stored in compress sparse row (CSR) format
// n: number of data points (length of ID array)
// d: dimension/number of coordinates of a data point
// nnz: number of nonzero coordinates

// ---------
// PARAMETER
// ---------
// level: number of tree level
// nTree: number of iterations for the randomized-KNN algorithm
// k: number of nearest neighbors to compute
// blkPoint: number of points in a batch for the leaf kernel
// cores: number of cores/threads used

// ------
// OUTPUT
// ------
// nborID: IDs of KNN for all data points; n-by-k matrix stored in row major
// nborDist: distances of KNN for all data points; n-by-k matrix in row major


#endif
