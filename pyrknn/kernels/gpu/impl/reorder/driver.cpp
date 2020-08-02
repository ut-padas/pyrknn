#include <iostream>
#include <random>

#include <Eigen/Sparse>
typedef Eigen::SparseMatrix<float,Eigen::RowMajor> SpMat; // row-major sparse matrix
typedef Eigen::Triplet<float> T;

#include <Eigen/Dense>
typedef Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> Mat;

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


void gather_gpu(int*, int*, float*, int, int, int, int*);

void scatter_gpu(float*, int, int, int*);

void gather_gpu(float*, int, int, const int*);

void test_gather() {
  int m = 9, n = 7;
  Mat A = Mat::Random(m, n);
  Eigen::PermutationMatrix<Eigen::Dynamic,Eigen::Dynamic> perm(m);
  perm.setIdentity();
  std::random_shuffle(perm.indices().data(), perm.indices().data()+m);
  Mat B = perm.transpose()*A;
  gather_gpu(A.data(), m, n, perm.indices().data());
  std::cout<<"Error of gather(): "<<(B-A).norm()<<std::endl;
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

  Eigen::PermutationMatrix<Eigen::Dynamic,Eigen::Dynamic> perm(m);
  perm.setIdentity();
  std::random_shuffle(perm.indices().data(), perm.indices().data()+m);
  
  //std::cout<<"A:\n"<<A<<"\n"
    //<<"P:\n"<<perm.indices()<<std::endl;

  Mat B = perm.transpose() * A;

  gather_gpu(A.outerIndexPtr(), A.innerIndexPtr(), A.valuePtr(), m, n, A.nonZeros(), 
      perm.indices().data());

  /*
  std::cout<<"B cpu:\n"<<B<<"\n"
    <<"B gpu:\n"<<A
    <<std::endl;
  */

  std::cout<<"Error of gather: "<<(B-A).norm()<<"\n";

  //std::cout<<"B:\n"<<B<<std::endl;
  Mat C = perm * B;

  scatter_gpu(B.data(), m, n, perm.indices().data());

  //std::cout<<"C cpu:\n"<<C<<"\n"
    //<<"C gpu:\n"<<B
    //<<std::endl;
  std::cout<<"Error of scatter: "<<(C-B).norm()<<"\n";

  test_gather();

  return 0;
}

