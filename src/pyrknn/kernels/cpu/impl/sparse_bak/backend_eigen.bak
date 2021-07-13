#include "NLA.hpp"
#include "timer.hpp"

#include <Eigen/Sparse>
#include <Eigen/Dense>
#include "util_eigen.hpp"

#include <algorithm>
#include <iostream>

typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> EMat;
typedef Eigen::VectorXf EVec;
typedef Eigen::SparseMatrix<float,Eigen::RowMajor> SpMat; // row-major sparse matrix


void orthonormalize(fMatrix &A) {
  unsigned m = A.rows();
  unsigned n = A.cols();

  auto copy = Eigen::Map<EMat>(A.data(), m, n);
  Eigen::HouseholderQR<EMat> qr(copy);

  EMat Q = EMat::Identity(m, n);
  Q = qr.householderQ() * Q;

  std::copy(Q.data(), Q.data()+m*n, A.data());
}


void GEMM_SDD(Points &P, fMatrix &R, fMatrix &X) {
  GEMM_SDD(P.rows(), R.cols(), P.cols(), P.rowPtr, P.colIdx, P.val, P.nonZeros(),
      R.data(), X.data());
}


void gather(Points &P, const ivec &perm) {
  auto A = Eigen::MappedSparseMatrix<float, Eigen::RowMajor>
              (P.n, P.d, P.nnz, P.rowPtr, P.colIdx, P.val);
  Eigen::PermutationMatrix<Eigen::Dynamic> B(P.n);
  std::copy(perm.data(), perm.data()+perm.size(), B.indices().data());
  SpMat C = B.transpose() * A; // note the transpose

  assert(C.rows() == P.n);
  assert(C.cols() == P.d);
  assert(C.nonZeros() == P.nnz);
  std::copy(C.outerIndexPtr(), C.outerIndexPtr()+P.n+1, P.rowPtr);
  std::copy(C.innerIndexPtr(), C.innerIndexPtr()+P.nnz, P.colIdx);
  std::copy(C.valuePtr(), C.valuePtr()+P.nnz, P.val);
}


void scatter(Points &P, const ivec &perm) {
  auto A = Eigen::MappedSparseMatrix<float, Eigen::RowMajor>
              (P.n, P.d, P.nnz, P.rowPtr, P.colIdx, P.val);
  Eigen::PermutationMatrix<Eigen::Dynamic> B(P.n);
  std::copy(perm.data(), perm.data()+perm.size(), B.indices().data());
  SpMat C = B * A;

  assert(C.rows() == P.n);
  assert(C.cols() == P.d);
  assert(C.nonZeros() == P.nnz);
  std::copy(C.outerIndexPtr(), C.outerIndexPtr()+P.n+1, P.rowPtr);
  std::copy(C.innerIndexPtr(), C.innerIndexPtr()+P.nnz, P.colIdx);
  std::copy(C.valuePtr(), C.valuePtr()+P.nnz, P.val);
}


// distance is symmetric
void compute_distance(const Points &P, fMatrix &Dt, double &t) {
  Timer timer;
  auto A = Eigen::MappedSparseMatrix<float, Eigen::RowMajor>
              (P.n, P.d, P.nnz, P.rowPtr, P.colIdx, P.val);
  EVec nrm = rowNorm(A);

  timer.start();
  EMat D = -2*A*A.transpose();
  timer.stop(); t += timer.elapsed_time();
  
  D.colwise() += nrm;
  D.rowwise() += nrm.transpose();
 
  std::copy(D.data(), D.data()+P.n*P.n, Dt.data());
  //std::cout<<"Norm:\n"<<nrm<<std::endl;
}


void orthonormal_bases(float *data, unsigned m, unsigned n) {
  auto A = Eigen::Map<EMat>(data, m, n);
  Eigen::HouseholderQR<EMat> qr(A);

  EMat thinQ = EMat::Identity(m, n);
  thinQ = qr.householderQ() * thinQ;

  std::copy(thinQ.data(), thinQ.data()+m*n, data);

#if 0
  std::cout<<"A:\n"<<A.topLeftCorner(2,2)<<std::endl;
  std::cout<<"qr error: "<<(thinQ.transpose()*thinQ-Mat::Identity(n,n)).norm()<<std::endl;
#endif
}


void GEMM_SDD(unsigned m, unsigned n, unsigned k, int *rowPtr, int *colIdx, float *val, 
    unsigned nnz, float *R, float *X) {

  auto A = Eigen::MappedSparseMatrix<float, Eigen::RowMajor>
              (m, k, nnz, rowPtr, colIdx, val);
  auto B = Eigen::Map<EMat>(R, k, n);
  
  EMat  C = A*B;
  std::copy(C.data(), C.data()+m*n, X);
  //std::cout<<"# threads used by Eigen: "<<Eigen::nbThreads()<<std::endl;
}


