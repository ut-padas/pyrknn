#include <iostream>
#include <random>
#include <vector>
#include <numeric> // std::iota

#include "spknn.hpp"

#include <Eigen/Sparse>
typedef Eigen::SparseMatrix<float,Eigen::RowMajor> SpMat; // row-major sparse matrix
typedef Eigen::Triplet<float> T;

#include <Eigen/Dense>
typedef Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> Mat;
typedef Eigen::Matrix<int,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> MatInt;
typedef Eigen::VectorXf Vec;


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


void compute_distance(const SpMat &A, Mat &D) {
  Mat X = A;
  D = -2*X*X.transpose();
  Vec nrm = X.rowwise().squaredNorm();
  D.rowwise() += nrm.transpose();
  D.colwise() += nrm;
  //std::cout<<"point norm:\n"<<nrm<<std::endl;
}


int main(int argc, char* argv[]) {

  int n = 1024;
  int d = 64;
  int k = 64;
  int m = 64;
  int l = 3;
  int repeat = 5;
  bool debug = false;
  bool benchmark = false;
  float sparsity = 0.5;
  for (int i=1; i<argc; i++) {
    if (!strcmp(argv[i],"-n"))
      n = atoi(argv[i+1]);
    if (!strcmp(argv[i],"-d"))
      d = atoi(argv[i+1]);
    if (!strcmp(argv[i],"-k"))
      k = atoi(argv[i+1]);
    if (!strcmp(argv[i],"-l"))
      l = atoi(argv[i+1]);
    if (!strcmp(argv[i],"-m"))
      m = atoi(argv[i+1]);
    if (!strcmp(argv[i],"-r"))
      repeat = atoi(argv[i+1]);
    if (!strcmp(argv[i],"-D"))
      debug = true;
    if (!strcmp(argv[i],"-B"))
      benchmark = true;
    if (!strcmp(argv[i],"-sparse"))
      sparsity = atof(argv[i+1]);
  }
  assert(n>0);
  assert(d>0);
  assert(k>0);
  assert(m>0 && n%m == 0);
  assert(sparsity > 0.);
  std::cout.precision(2);
  std::cout<<std::scientific;
  std::cout<<"\n======================\n"
           <<"Inputs:\n"
           <<"----------------------\n"
           <<"N: "<<n<<std::endl
           <<"d: "<<d<<std::endl
           <<"k: "<<k<<std::endl
           <<"level: "<<l<<std::endl
           <<"sparsity: "<<sparsity<<std::endl
           <<"----------------------\n"
           <<"block size: "<<m<<std::endl
           <<"repeat: "<<repeat<<std::endl
           <<"debug: "<<debug<<std::endl
           <<"benchmark: "<<benchmark<<std::endl
           <<"----------------------\n"
           //<<"Mem (points): "<<4.*n*d*s*2*sparsity*2/1.e9<<" GB"<<std::endl
           //<<"Mem (dist etc.): "<<4.*n*m*s*4/1.e9<<" GB"<<std::endl
           <<"======================\n\n";

  // generate random sparse points
  SpMat P(n, d);
  P.reserve(Eigen::VectorXi::Constant(n, d*sparsity+3));
  init_random(P, n, d, sparsity);

  Mat nborDist(n, k);
  MatInt nborID(n, k);
  spknn(P.outerIndexPtr(), P.innerIndexPtr(), P.valuePtr(), n, d, P.nonZeros(), l,
      nborID.data(), nborDist.data(), k, m);


  return 0;
}

int test_kernel(int argc, char* argv[]) {

  int n = 1024;
  int d = 64;
  int m = 64;
  int s = 10;
  int repeat = 5;
  bool debug = false;
  bool benchmark = false;
  float sparsity = 0.1;
  for (int i=1; i<argc; i++) {
    if (!strcmp(argv[i],"-n"))
      n = atoi(argv[i+1]);
    if (!strcmp(argv[i],"-d"))
      d = atoi(argv[i+1]);
    if (!strcmp(argv[i],"-s"))
      s = atoi(argv[i+1]);
    if (!strcmp(argv[i],"-m"))
      m = atoi(argv[i+1]);
    if (!strcmp(argv[i],"-r"))
      repeat = atoi(argv[i+1]);
    if (!strcmp(argv[i],"-D"))
      debug = true;
    if (!strcmp(argv[i],"-B"))
      benchmark = true;
    if (!strcmp(argv[i],"-sparse"))
      sparsity = atof(argv[i+1]);
  }
  assert(n>0);
  assert(d>0);
  assert(s>0);
  assert(m>0 && n%m==0);
  assert(sparsity > 0.);
  std::cout.precision(2);
  std::cout<<std::scientific;
  std::cout<<"\n======================\n"
           <<"Inputs:\n"
           <<"----------------------\n"
           <<"N: "<<n<<std::endl
           <<"d: "<<d<<std::endl
           <<"num_leaf: "<<s<<std::endl
           <<"sparsity: "<<sparsity<<std::endl
           <<"----------------------\n"
           <<"block size: "<<m<<std::endl
           <<"repeat: "<<repeat<<std::endl
           <<"debug: "<<debug<<std::endl
           <<"benchmark: "<<benchmark<<std::endl
           <<"----------------------\n"
           <<"Mem (points): "<<4.*n*d*s*2*sparsity*2/1.e9<<" GB"<<std::endl
           <<"Mem (dist etc.): "<<4.*n*m*s*4/1.e9<<" GB"<<std::endl
           <<"======================\n\n";

  // y = P * x
  std::vector<SpMat> P(s); 
  std::vector<Vec> x(s), yCPU(s), yGPU(s);
  for (int i=0; i<s; i++) {
    P[i].resize(n, d); // TODO: different number of rows
    P[i].reserve(Eigen::VectorXi::Constant(n, d*sparsity+3));
    init_random(P[i], n, d, sparsity);
    x[i] = Vec::Random(d);
    yCPU[i] = P[i]*x[i];
    yGPU[i].resize(n);
   
    if (debug) {
    std::cout<<"P["<<i<<"]:\n"<<P[i]<<std::endl
             <<"x["<<i<<"]:\n"<<x[i]<<std::endl
             <<"y["<<i<<"]:\n"<<yCPU[i]<<std::endl;
    }
  }
  
  
  // sequential kernel launch
  double err = 0., norm = 0.;
  for (int i=0; i<s; i++) {
    gemv_gpu(P[i].outerIndexPtr(), P[i].innerIndexPtr(), P[i].valuePtr(), 
        P[i].rows(), P[i].cols(), P[i].nonZeros(),
        x[i].data(), yGPU[i].data());
    err += (yGPU[i] - yCPU[i]).norm();
    norm += yCPU[i].norm();
  }
  std::cout<<"Error of GEMV: "<<err/norm<<std::endl;


  // batched gemv()
  int *rowPtr[s], *colIdx[s], M[s], NNZ[s];
  float *val[s], *xPtr[s], *yPtr[s];
  for (int i=0; i<s; i++) {
    rowPtr[i] = P[i].outerIndexPtr();
    colIdx[i] = P[i].innerIndexPtr();
    val[i] = P[i].valuePtr();
    M[i] = P[i].rows();
    NNZ[i] = P[i].nonZeros();
    xPtr[i] = x[i].data();
    yPtr[i] = yGPU[i].data();
  }
  batchedGemv_gpu(rowPtr, colIdx, val, M, d, NNZ, xPtr, yPtr, s);

  err = 0.;
  for (int i=0; i<s; i++) {
    err += (yGPU[i] - yCPU[i]).norm();
    if (debug)
    std::cout<<"yGPU["<<i<<"]:\n"<<yGPU[i]<<std::endl;
  }
  std::cout<<"Error of batched GEMV: "<<err/norm<<std::endl;


  //-------------------------------------------
  // compute distance
  std::vector<Mat> DCPU(s);
  for (int i=0; i<s; i++) {
    compute_distance(P[i], DCPU[i]);
  }

  Mat D(n*s, n);
  compute_distance_gpu(rowPtr, colIdx, val, M, d, NNZ, D.data(), s, m, debug);
  
  err = 0., norm = 0.;
  for (int i=0; i<s; i++) {
    err += (DCPU[i]-D.middleRows(i*n, n)).norm();
    norm += DCPU[i].norm();
  }
  std::cout<<"Error of batched GEMM: "<<err/norm<<std::endl;

  if (debug) {
    std::cout<<"CPU:\n"<<std::endl;
    for (int i=0; i<s; i++)
      std::cout<<"D["<<i<<"]:\n"<<DCPU[i]<<std::endl;
    
    std::cout<<"GPU:\n"<<D<<std::endl;
  }

  return 0;
}

