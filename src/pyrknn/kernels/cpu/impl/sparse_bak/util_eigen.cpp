#include "util_eigen.hpp"

#include <Eigen/Sparse>
typedef Eigen::SparseMatrix<float,Eigen::RowMajor> SpMat; // row-major sparse matrix

#include <Eigen/Dense>
typedef Eigen::VectorXf Vec;


Vec rowNorm(const SpMat &A) {
  //Timer t; t.start();
  //SpMat copy = A.cwiseProduct(A);
  SpMat copy = A;
  float *val = copy.valuePtr();
  for (int i=0; i<copy.nonZeros(); i++)
    val[i] = val[i]*val[i];
  //t.stop(); std::cout<<"[CPU] row norm: "<<t.elapsed_time()<<" s\n";
  
  Vec A2 = copy * Vec::Constant(A.cols(), 1.0);
  return A2;
}


void print(const Points &P, std::string name) {
  auto A = Eigen::MappedSparseMatrix<float, Eigen::RowMajor>
              (P.n, P.d, P.nnz, P.rowPtr, P.colIdx, P.val);
  std::cout<<name<<":\n"<<A<<std::endl;
}

