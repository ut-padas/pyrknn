#include "readSVM.hpp"
#include "timer.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <assert.h>


int reverseInt (int i) 
{
    unsigned char c1, c2, c3, c4;

    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;

    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

Mat read_mnist() {
  std::ifstream file("/scratch/06108/chaochen/train-images-idx3-ubyte");
  assert(file.good());
  int magic, nImage, nrow, ncol;
  file.read((char*)&magic, sizeof(int));
  file.read((char*)&nImage, sizeof(int));
  file.read((char*)&nrow, sizeof(int));
  file.read((char*)&ncol, sizeof(int));
  magic = reverseInt(magic);
  nImage = reverseInt(nImage);
  nrow = reverseInt(nrow);
  ncol = reverseInt(ncol);
  std::cout<<"magic number: "<<magic<<"\n";
  std::cout<<nImage<<" images of "<<nrow<<" rows and "<<ncol<<" columns.\n";
  Mat A(nImage, nrow*ncol);
  char c;
  for (int i=0; i<nImage; i++) {
    for (int j=0; j<nrow*ncol; j++) {
      file.read(&c, sizeof(c));
      A(i,j) = c;
    }
  }
  return A;
}


SpMat read_svm(const std::vector<std::string> &files, int ncol) {

  int lable; // dummy variable
  int col;
  char colon; // separator
  float val;

  std::string line;
  typedef Eigen::Triplet<float> T;
  std::vector<T> tripletList;

  int row = 0;
  for (auto filename: files) {
    std::ifstream ifile(filename);
    while (std::getline(ifile, line)) {
      std::istringstream ss(line);
      ss >> lable;
      while (ss >> col >> colon >> val) {
        tripletList.push_back( T(row,col-1,val) );
      }
      row++;
    }
    ifile.close();
  }
  
  SpMat A(row, ncol);
  A.setFromTriplets(tripletList.begin(), tripletList.end());
  A.makeCompressed();
  return A;
}


SpMat read_kdd12_dataset() {
  std::vector<std::string> files;
  std::string file1("/scratch/06108/chaochen/kdd12");
  files.push_back(file1);
  return read_svm(files, 54686452);
}


SpMat read_criteo_dataset() {
  std::vector<std::string> files;
  std::string file1("/scratch/06108/chaochen/criteo.kaggle2014.train.svm");
  files.push_back(file1);
  return read_svm(files, 1000000);
}


SpMat read_avazu_dataset() {
  std::vector<std::string> files;
  //std::string file1("/scratch/06108/chaochen/avazu-app.t");
  std::string file2("/scratch/06108/chaochen/avazu-app");
  //std::string file3("/scratch/06108/chaochen/avazu-site");
  //files.push_back(file1);
  files.push_back(file2);
  //files.push_back(file3);
  return read_svm(files, 1000000);
}


SpMat read_url_dataset(int nDays) {
  assert(nDays>0 && nDays<122);
  int nfile = nDays;
  std::vector<std::string> files(nfile);
  std::string dir("/scratch/06108/chaochen/url_svmlight/Day");
  for (int i=0; i<nfile; i++)
    files[i] = dir + std::to_string(i) + ".svm";

  return read_svm(files, 3231961);
}


void write_csr_binary(const SpMat &A, std::string filename) {
  std::ofstream ofile(filename, std::ios::out | std::ios::binary);
  int m = A.rows();
  int n = A.cols();
  int nnz = A.nonZeros();
  ofile.write((char *)&m, sizeof(int));
  ofile.write((char *)&n, sizeof(int));
  ofile.write((char *)&nnz, sizeof(int));
  const int *rowPtr = A.outerIndexPtr();
  ofile.write((char *)rowPtr, (A.rows()+1)*sizeof(int));
  const int *colIdx = A.innerIndexPtr();
  ofile.write((char *)colIdx, A.nonZeros()*sizeof(int));
  const float *val = A.valuePtr();
  ofile.write((char *)val, A.nonZeros()*sizeof(float));
  ofile.close();
}


SpMat read_csr_binary(std::string filename) {
  std::ifstream ifile(filename, std::ios::in | std::ios::binary);
  int m, n, nnz;
  ifile.read((char *)&m, sizeof(int));
  ifile.read((char *)&n, sizeof(int));
  ifile.read((char *)&nnz, sizeof(int));
  int *rowPtr = new int[m+1];
  ifile.read((char *)rowPtr, (m+1)*sizeof(int));
  int *colIdx = new int[nnz];
  ifile.read((char *)colIdx, nnz*sizeof(int));
  float *val = new float[nnz];
  ifile.read((char *)val, nnz*sizeof(float));
  ifile.close();
  SpMat A = Eigen::MappedSparseMatrix<float, Eigen::RowMajor>
              (m, n, nnz, rowPtr, colIdx, val);
  delete[] rowPtr;
  delete[] colIdx;
  delete[] val;
  return A;
}


void write_csr(const SpMat &A, std::string filename) {
  std::ofstream ofile(filename);
  ofile<<A.rows()<<" "<<A.cols()<<" "<<A.nonZeros()<<std::endl;
  const int *rowPtr = A.outerIndexPtr();
  for (int i=0; i<A.rows()+1; i++)
    ofile<<rowPtr[i]<<" ";
  ofile<<std::endl;
  const int *colIdx = A.innerIndexPtr();
  for (int i=0; i<A.nonZeros(); i++)
    ofile<<colIdx[i]<<" ";
  ofile<<std::endl;
  const float *val = A.valuePtr();
  for (int i=0; i<A.nonZeros(); i++)
    ofile<<val[i]<<" ";
  ofile<<std::endl;
  ofile.close();
}


SpMat read_csr(std::string filename) {
  int m, n, nnz;
  std::ifstream ifile(filename);
  ifile>>m>>n>>nnz;
  int *rowPtr = new int[m+1];
  for (int i=0; i<m+1; i++)
    ifile>>rowPtr[i];
  int *colIdx = new int[nnz];
  for (int i=0; i<nnz; i++)
    ifile>>colIdx[i];
  float *val = new float[nnz];
  for (int i=0; i<nnz; i++)
    ifile>>val[i];
  ifile.close();
  SpMat A = Eigen::MappedSparseMatrix<float, Eigen::RowMajor>
              (m, n, nnz, rowPtr, colIdx, val);
  delete[] rowPtr;
  delete[] colIdx;
  delete[] val;
  return A;
}

