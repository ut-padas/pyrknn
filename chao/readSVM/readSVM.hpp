#ifndef READ_SVM_HPP
#define READ_SVM_HPP

#include <Eigen/Sparse>

typedef Eigen::SparseMatrix<float,Eigen::RowMajor> SpMat; // row-major sparse matrix

SpMat read_url_dataset(int);
SpMat read_avazu_dataset();
SpMat read_criteo_dataset();
SpMat read_kdd12_dataset();

void write_csr(const SpMat&, std::string);

SpMat read_csr(std::string);

void write_csr_binary(const SpMat&, std::string);

SpMat read_csr_binary(std::string);

#endif
