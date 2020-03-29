#include <iostream>
#include <random>

#include "transpose.hpp"

#include <Eigen/Sparse>
typedef Eigen::SparseMatrix<float,Eigen::RowMajor> SpMat; // row-major sparse matrix
typedef Eigen::Triplet<float> T;


std::default_random_engine gen;
void init_random(SpMat &A, int M, int N, float sparsity) {
  std::uniform_real_distribution<float> dist(0.0,1.0);
  std::vector<T> tripletList;
  for(int i=0; i<M; ++i) {
    for(int j=0; j<N; ++j) {
       auto x = dist(gen);
       if (x < sparsity) {
           tripletList.push_back( T(i,j,x) );
       }
    }
  }
  A.setFromTriplets(tripletList.begin(), tripletList.end());
  A.makeCompressed();
}


int main(int argc, char *argv[]) {
  
  int m = 5;
  int n = 3;
  float sparsity = 0.6;
  for (int i=1; i<argc; i++) {
    if (!strcmp(argv[i],"-m"))
      m = atoi(argv[i+1]);
    if (!strcmp(argv[i],"-n"))
      n = atoi(argv[i+1]);
    if (!strcmp(argv[i],"-s"))
      sparsity = atof(argv[i+1]);
  }  

 std::cout<<"\n======================\n"
           <<"Inputs:\n"
           <<"----------------------\n"
           <<"m: "<<m<<std::endl
           <<"n: "<<n<<std::endl
           <<"sparsity: "<<sparsity<<std::endl
           <<"======================\n\n";

  SpMat A(m, n);
  init_random(A, m, n, sparsity);

  int nnz = A.nonZeros();
  int rowPtr[n+1], colIdx[nnz];
  float val[nnz];
  transpose(m, n, nnz, A.outerIndexPtr(), A.innerIndexPtr(), A.valuePtr(),
      rowPtr, colIdx, val);
  auto T = Eigen::MappedSparseMatrix<float, Eigen::RowMajor>
                (n, m, nnz, rowPtr, colIdx, val);

  SpMat At = A.transpose();
 
  if (m<10 && n<10) { 
    std::cout<<"A:\n"<<A<<"\n";
             //<<"Transpose:\n"<<At<<"\n";

    std::cout<<"T:\n"<<T<<"\n";
  }
  std::cout<<"Error: "<<(T-At).norm()<<"\n";

  return 0;
}

