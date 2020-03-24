#include "util.hpp"

#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <iostream>

using namespace Eigen;
typedef Matrix<float, Dynamic, Dynamic, RowMajor> Mat;
typedef Matrix<int, Dynamic, Dynamic, RowMajor> MatInt;
typedef VectorXf Vec;

void print(int m, int n, int nnz, int *rowPtr, int *colIdx, float *val, const std::string &name) {
  auto A = Eigen::MappedSparseMatrix<float, Eigen::RowMajor>
          (m, n, nnz, rowPtr, colIdx, val);
  std::cout<<"\n"<<name<<":\n"<<A<<std::endl;
}


void print(int m, int n, float *val, const std::string &name) {
  auto A = Eigen::Map<Mat>(val, m, n);
  std::cout<<"\n"<<name<<":\n"<<A<<std::endl;
}


void dprint(int m, int *dval, const std::string &name) {
  int hval[m];
  copy(m, dval, hval);
  auto A = Eigen::Map<VectorXi>(hval, m);
  std::cout<<"\n"<<name<<":\n"<<A.transpose()<<std::endl;
}


void dprint(int m, float *dval, const std::string &name) {
  float hval[m];
  copy(m, dval, hval);
  auto A = Eigen::Map<Vec>(hval, m);
  std::cout<<"\n"<<name<<":\n"<<A.transpose()<<std::endl;
}


void dprint(int m, int n, int *dval, const std::string &name) {
  int hval[m*n];
  copy(m*n, dval, hval);
  auto A = Eigen::Map<MatInt>(hval, m, n);
  std::cout<<"\n"<<name<<":\n"<<A<<std::endl;
}


void dprint(int m, int n, float *dval, const std::string &name) {
  float hval[m*n];
  copy(m*n, dval, hval);
  auto A = Eigen::Map<Mat>(hval, m, n);
  std::cout<<"\n"<<name<<":\n"<<A<<std::endl;
}


void dprint(int m, int n, int nnz, int *dRowPtr, int *dColIdx, float *dVal, 
    const std::string &name) {
  // copy data to host
  int hRowPtr[m+1], hColIdx[nnz];
  float hVal[nnz];
  copy_spmat_d2h(m, n, nnz, dRowPtr, dColIdx, dVal, hRowPtr, hColIdx, hVal);
  // form the sparse matrix
  auto A = Eigen::MappedSparseMatrix<float, Eigen::RowMajor>
          (m, n, nnz, hRowPtr, hColIdx, hVal);
  std::cout<<"\n"<<name<<":\n"<<A<<std::endl;
}

