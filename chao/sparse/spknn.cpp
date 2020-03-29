#include <iostream>
#include <random>
#include <vector>
#include <numeric> // std::iota

#include "spknn.hpp"
#include "rand.hpp"
#include "timer.hpp"
#include "readSVM.hpp"

#include <Eigen/Sparse>
typedef Eigen::SparseMatrix<float,Eigen::RowMajor> SpMat; // row-major sparse matrix
typedef Eigen::Triplet<float> T;

#include <Eigen/Dense>
typedef Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> Mat;
typedef Eigen::Matrix<int,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> MatInt;
typedef Eigen::VectorXf Vec;


std::default_random_engine gen;
void init_random(SpMat &A, int M, int N, float sparsity) {
  //std::uniform_real_distribution<float> dist(0.0,1.0);
  float *val = new float[M*N];
  init_random_gpu(val, M*N);
  std::vector<T> tripletList;
  for(int i=0; i<M; ++i) {
    for(int j=0; j<N; ++j) {
       auto x = val[i*N+j];
       if (x < sparsity) {
           tripletList.push_back( T(i,j,x) );
       }
    }
  }
  A.setFromTriplets(tripletList.begin(), tripletList.end());
  A.makeCompressed();
  delete[] val;
}


void compute_distance(const SpMat &A, Mat &D) {
  Mat X = A;
  D = -2*X*X.transpose();
  Vec nrm = X.rowwise().squaredNorm();
  D.rowwise() += nrm.transpose();
  D.colwise() += nrm;
  //std::cout<<"point norm:\n"<<nrm<<std::endl;
}


//int test_dataset(int argc, char* argv[]) {
int main(int argc, char* argv[]) {
  
  int nDays = 1;
  int level = 5;
  int k = 3;
  int nRow = -1;
  for (int i=1; i<argc; i++) {
    if (!strcmp(argv[i],"-days"))
      nDays = atoi(argv[i+1]);
    if (!strcmp(argv[i],"-l"))
      level = atoi(argv[i+1]);
    if (!strcmp(argv[i],"-k"))
      k = atoi(argv[i+1]);
    if (!strcmp(argv[i],"-row"))
      nRow = atoi(argv[i+1]);
  }
  std::cout<<"\n======================\n"
           <<"Inputs:\n"
           <<"----------------------\n"
           <<"# days: "<<nDays<<std::endl
           <<"tree level: "<<level<<std::endl
           <<"k: "<<k<<std::endl
           //<<"rows: "<<nRow<<std::endl
           <<"======================\n\n";
  assert(nDays > 0);
  assert(level > 0);
  assert(k > 0);

  Timer t; t.start();
  //SpMat P = read_url_dataset(nDays);
  SpMat P = read_csr_binary("/scratch/06108/chaochen/url_csr.bin");
  //SpMat P = read_csr_binary("/scratch/06108/chaochen/avazu_csr.bin");
  if (nRow > 0) P = P.topRows(nRow); // for debugging
  t.stop();

  std::cout<<"\n# rows: "<<P.rows()<<"\n"
           <<"# columns: "<<P.cols()<<"\n"
           <<"# nonzeros: "<<P.nonZeros()<<"\n"
           <<"time for reading data: "<<t.elapsed_time()<<" s\n";
 
  int N = P.rows();
  Mat nborDist(N, k);
  MatInt nborID(N, k);

  t.start();
  spknn(P.outerIndexPtr(), P.innerIndexPtr(), P.valuePtr(), P.rows(), P.cols(), P.nonZeros(), 
      level, nborID.data(), nborDist.data(), k);
  t.stop();


  // output some results
  std::cout<<"\t*** First 3 points ***\n"
           <<"neighbor ID:\n"<<nborID.topRows(3)<<"\n"
           <<"neighbor distance:\n"<<nborDist.topRows(3)<<"\n";
    
  std::cout<<"\n SPKNN time: "<<t.elapsed_time()<<" s\n\n";

  return 0;
}


int test_random_data(int argc, char *argv[]) {
//int main(int argc, char *argv[]) {

  int n = 1024; // points per leaf node
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
  int nLeaf = 1<<(l-1);
  int N = n*nLeaf;
  std::cout.precision(2);
  std::cout<<std::scientific;
  std::cout<<"\n======================\n"
           <<"Inputs:\n"
           <<"----------------------\n"
           <<"N: "<<N<<std::endl
           <<"d: "<<d<<std::endl
           <<"k: "<<k<<std::endl
           <<"level: "<<l<<std::endl
           <<"sparsity: "<<sparsity<<std::endl
           //<<"num leaf: "<<nLeaf<<std::endl
           //<<"leaf size: "<<N/nLeaf<<std::endl
           <<"----------------------\n"
           <<"block size: "<<m<<std::endl
           <<"repeat: "<<repeat<<std::endl
           <<"debug: "<<debug<<std::endl
           <<"benchmark: "<<benchmark<<std::endl
           <<"----------------------\n"
           <<"Mem (points): "<<4.*N*d*sparsity*2/1.e9<<" GB"<<std::endl
           <<"Mem (dist block): "<<4.*n*m*nLeaf*4/1.e9<<" GB"<<std::endl
           <<"Mem (output): "<<4.*N*k*2/1.e9<<" GB"<<std::endl
           <<"======================\n\n";
  assert(n>0);
  assert(d>0);
  assert(k>0);
  assert(m>0);
  assert(sparsity > 0.);

  // generate random sparse points
  SpMat P(N, d);
  P.reserve(Eigen::VectorXi::Constant(N, d*sparsity+3));

  float t0, t1;
  Timer t; t.start();
  init_random(P, N, d, sparsity);
  t.stop(); t0 = t.elapsed_time();
  if (debug) std::cout<<"Points:\n"<<P<<std::endl;

  Mat nborDist(N, k);
  MatInt nborID(N, k);

  t.start();
  spknn(P.outerIndexPtr(), P.innerIndexPtr(), P.valuePtr(), N, d, P.nonZeros(), l,
      nborID.data(), nborDist.data(), k, m);
  t.stop(); t1 = t.elapsed_time();

  std::cout<<"\nTime for generating points: "<<t0<<" s\n"
           <<"Time for sparse KNN: "<<t1<<" s"<<std::endl;

  return 0;
}

/*
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
*/
