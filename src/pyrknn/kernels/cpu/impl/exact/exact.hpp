#ifndef EXACT_HPP
#define EXACT_HPP

#include "mpi.h"
#include <stdio.h>

#define CHECK_MPI(func)                                                       \
{                                                                             \
    int status = (func);                                                      \
    if (status != MPI_SUCCESS) {                                              \
      int eclass, len;                                                             \
      char estring[MPI_MAX_ERROR_STRING];                                     \
      MPI_Error_class(status, &eclass);                                        \
      MPI_Error_string(status, estring, &len);                                 \
      printf("MPI API failed at line %d with error %d: %s\n",                 \
               __LINE__, eclass, estring);                                    \
        fflush(stdout);                                                       \
    }                                                                         \
}


#include <Eigen/Sparse>
typedef Eigen::SparseMatrix<float,Eigen::RowMajor> SpMat; // row-major sparse matrix
typedef Eigen::Triplet<float> T;

#include <Eigen/Dense>
typedef Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> Mat;
typedef Eigen::Matrix<int,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> MatInt;
typedef Eigen::VectorXf Vec;
typedef Eigen::VectorXi VecInt;


void exact_knn
(int nQ, int dQ, int nnzQ, int *rowPtrQ, int *colIdxQ, float *valQ,
  int nR, int dR, int nnzR, int *rowPtrR, int *colIdxR, float *valR,
  int *ID, int k, int *nborID, float *nborDist);


void merge(int, int, int*, float*);


#endif
