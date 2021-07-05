#include "exact.hpp"

#include <assert.h>
#include <vector>
#include <numeric>
#include <algorithm>


Vec row_norm(const SpMat &A) {
  SpMat copy = A;
  float *val = copy.valuePtr();
  for (int i=0; i<copy.nonZeros(); i++)
    val[i] = val[i]*val[i];
  
  Vec A2 = copy * Vec::Constant(A.cols(), 1.0);
  return A2;
}

void kselect(const float *value, const int *ID, int n, float *kval, int *kID, int k) {
  std::vector<int> idx(n);
  std::iota(idx.begin(), idx.end(), 0);
  std::stable_sort(idx.begin(), idx.end(),
      [&value](int i, int j) {return value[i]<value[j];});
  for (int i=0; i<k; i++) {
    int j = idx[i];
    kval[i] = value[j];
    kID[i] = ID[j];
  }
}

void exact_knn
(int nQ, int dQ, int nnzQ, int *rowPtrQ, int *colIdxQ, float *valQ,
 int nR, int dR, int nnzR, int *rowPtrR, int *colIdxR, float *valR,
 int *ID, int k, int *nborID, float *nborDist) {
  assert(dQ==dR);
  SpMat Q = Eigen::MappedSparseMatrix<float, Eigen::RowMajor>
              (nQ, dQ, nnzQ, rowPtrQ, colIdxQ, valQ);
  SpMat R = Eigen::MappedSparseMatrix<float, Eigen::RowMajor>
              (nR, dR, nnzR, rowPtrR, colIdxR, valR);
  // compute distance
  Vec Q2 = row_norm(Q);
  Vec R2 = row_norm(R);
  //Mat D2 = -2*( Q*R.transpose() ).pruned();
  Mat D2 = -2 * Q * R.transpose();
  D2.colwise() += Q2;
  D2.rowwise() += R2.transpose();
  // find neighbor
  for (int i=0; i<nQ; i++) {
    kselect(D2.data()+i*nR, ID, nR, nborDist+i*k, nborID+i*k, k);
  }
}


typedef Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> CMat; // row-major in C
typedef Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::ColMajor> FMat; // column-major in Fortran
typedef Eigen::Matrix<int,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> CMatInt;
typedef Eigen::Matrix<int,Eigen::Dynamic,Eigen::Dynamic,Eigen::ColMajor> FMatInt;

void merge(int n, int k, int *key, float *val) {
  CMatInt ID = Eigen::Map<CMatInt>(key, n, k);
  FMatInt ID_send = ID;
  CMat    Dist = Eigen::Map<CMat>(val, n, k);
  FMat    Dist_send = Dist;

  int nproc, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  FMatInt ID_recv(n, k*nproc);
  CHECK_MPI( MPI_Gather(ID_send.data(), n*k, MPI_INT, ID_recv.data(), n*k, MPI_INT, 
        0, MPI_COMM_WORLD) );
  FMat Dist_recv(n, k*nproc);
  CHECK_MPI( MPI_Gather(Dist_send.data(), n*k, MPI_FLOAT, Dist_recv.data(), n*k, MPI_FLOAT, 
        0, MPI_COMM_WORLD) );
 
  if (rank==0) {
    CMatInt ID_all = ID_recv;
    CMat    Dist_all = Dist_recv;
    for (int i=0; i<n; i++) {
      kselect(Dist_all.data()+i*k*nproc, ID_all.data()+i*k*nproc, k*nproc,
          val+i*k, key+i*k, k);
    }
  }
}


