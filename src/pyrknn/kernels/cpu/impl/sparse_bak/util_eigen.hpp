#ifndef util_eigen_hpp
#define util_eigen_hpp

#include <Eigen/Sparse>
#include <Eigen/Dense>

Eigen::VectorXf rowNorm(const Eigen::SparseMatrix<float, Eigen::RowMajor> &A);


#include "matrix.hpp"
void print(const Points &P, std::string);


#endif
