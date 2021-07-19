#include <iostream>
#include <random>

#include <Eigen/Sparse>
typedef Eigen::SparseMatrix<float, Eigen::RowMajor> SpMat; // row-major sparse matrix
typedef Eigen::Triplet<float> T;

#include <Eigen/Dense>
typedef Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic> Mat; // column-major

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


void gemm_ssd_gpu(int, int, int, int*, int*, float*, int, float*, float*);

void gemm_gpu(int, int, int, const float*, const float*, float*);

void test_gemm() {
  int m = 7, n = 5, k = 9;
  Mat A = Mat::Random(m, k), B = Mat::Random(k, n);
  Mat C(m, n);
  gemm_gpu(m, n, k, A.data(), B.data(), C.data());
  std::cout<<"Error of gemm(): "<<(C-A*B).norm()<<std::endl;
}

int main(int argc, char *argv[]) {
  
  int m = 5;
  int n = 4;
  int k = 3;
  float sparsity = 0.6;
  for (int i=1; i<argc; i++) {
    if (!strcmp(argv[i],"-m"))
      m = atoi(argv[i+1]);
    if (!strcmp(argv[i],"-n"))
      n = atoi(argv[i+1]);
    if (!strcmp(argv[i],"-k"))
      k = atoi(argv[i+1]);
    if (!strcmp(argv[i],"-s"))
      sparsity = atof(argv[i+1]);
  }  

 std::cout<<"\n======================\n"
           <<"Inputs:\n"
           <<"----------------------\n"
           <<"m: "<<m<<std::endl
           <<"n: "<<n<<std::endl
           <<"k: "<<k<<std::endl
           <<"sparsity: "<<sparsity<<std::endl
           <<"======================\n\n";

  SpMat A(m, k);
  init_random(A, m, k, sparsity);
  //A.setIdentity();

  Mat B = Mat::Random(k, n);
  //Mat B = Mat::Identity(k, n);
  Mat C = A * B;

  Mat C_gpu(m, n);
  gemm_ssd_gpu(m, n, k, A.outerIndexPtr(), A.innerIndexPtr(), A.valuePtr(), A.nonZeros(),
      B.data(), C_gpu.data());

  
  /*std::cout<<"A:\n"<<A<<"\n"
    <<"B:\n"<<B<<"\n"
    <<"C:\n"<<C<<"\n"
    <<"C_gpu:\n"<<C_gpu<<"\n";
    */
  std::cout<<"Error: "<<(C-C_gpu).norm()<<"\n";

  test_gemm();

  return 0;
}

