#include <iostream>

#include <vector>
#include <numeric> // std::iota
#include <algorithm> // std::stable_sort

#include <Eigen/Dense>

#include "knn_gpu.hpp"
#include "./util/timer.hpp"
#include "./util/rand.hpp"

using namespace Eigen;
typedef Matrix<float, Dynamic, Dynamic, RowMajor> Mat;
typedef Matrix<int, Dynamic, Dynamic, RowMajor> MatInt;
typedef VectorXf Vec;


Mat distSquared_cpu(const Mat& R, const Mat& Q, bool debug=false) {
  Vec R2 = R.rowwise().squaredNorm();
  Vec Q2 = Q.rowwise().squaredNorm();

  /*if (debug) {
    std::cout<<"R2:\n"<<R2<<std::endl
  	   <<"Q2:\n"<<Q2<<std::endl;
  }*/

  Mat D2 = -2*Q*R.transpose();
  D2.colwise() += Q2;
  D2.rowwise() += R2.transpose();
  //if (debug) std::cout<<"D2:\n"<<D2<<std::endl;
  return D2;
}


void kselect(const float *value, const int *ID, int n, float *kval, int *kID, int k) {
  std::vector<int> idx(n);
  std::iota(idx.begin(), idx.end(), 0);
  std::stable_sort(idx.begin(), idx.end(),
		  [&value](int i, int j) {return value[i]<value[j];});
  for (int i=0; i<k; i++) {
    int j = idx[i];
    kval[i] = value[j];
    kID[i] = ID[j];
  }
}


void knn_cpu(const Mat &R, const Mat &Q, const int *ID, int N, int d, 
    float *nborDist, int *nborID, int k, bool debug=false) {
  Mat D2 = distSquared_cpu(R, Q, debug);
  for (int i=0; i<N; i++)
    kselect(D2.data()+i*N, ID, N, nborDist+i*k, nborID+i*k, k);
}


int main(int argc, char* argv[]) {

  int n = 1024;
  int d = 64;
  int k = 64;
  int s = 1;
  int m = 64;
  int repeat = 5;
  bool debug = false;
  bool benchmark = false;
  for (int i=1; i<argc; i++) {
    if (!strcmp(argv[i],"-n"))
      n = atoi(argv[i+1]);
    if (!strcmp(argv[i],"-d"))
      d = atoi(argv[i+1]);
    if (!strcmp(argv[i],"-k"))
      k = atoi(argv[i+1]);
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
  }
  assert(n>0);
  assert(k>0);
  assert(d>0);
  assert(s>0);
  assert(m>0 && n%m==0);
  std::cout.precision(2);
  std::cout<<std::scientific;
  std::cout<<"\n======================\n"
           <<"Inputs:\n"
           <<"----------------------\n"
           <<"N: "<<n<<std::endl
           <<"d: "<<d<<std::endl
           <<"k: "<<k<<std::endl
           <<"num_leaf: "<<s<<std::endl
           <<"----------------------\n"
           <<"block size: "<<m<<std::endl
           <<"repeat: "<<repeat<<std::endl
           <<"debug: "<<debug<<std::endl
           <<"benchmark: "<<benchmark<<std::endl
           <<"----------------------\n"
           <<"Mem (points): "<<4.*n*d*s*2/1.e9<<" GB"<<std::endl
           <<"Mem (dist etc.): "<<4.*n*m*s*4/1.e9<<" GB"<<std::endl
           <<"Mem (nbor): "<<4.*n*k*s*2/1.e9<<" GB"<<std::endl
           <<"======================\n\n";

  const int nLeaf = s;
  const int N = n;
  double t_init = 0.;
  Timer t;

  // points
  Mat R(N*nLeaf, d), Q(N*nLeaf, d);
  float *R_ptr[nLeaf], *Q_ptr[nLeaf];

  // ID of R points
  std::vector<int> ID;
  int *ID_ptr[nLeaf];

  // initialize
  t.start();
  //R = Mat::Random(N*nLeaf, d);
  //Q = Mat::Random(N*nLeaf, d);
  init_random_gpu(R.data(), N*nLeaf*d);
  init_random_gpu(Q.data(), N*nLeaf*d);
  for (int i=0; i<nLeaf; i++) {
    R_ptr[i] = R.data()+i*N*d;
    Q_ptr[i] = Q.data()+i*N*d;
  }

  ID.resize(N*nLeaf);
  std::iota(ID.begin(), ID.end(), 0);
  for (int j=0; j<nLeaf; j++) {
    ID_ptr[j] = ID.data()+j*N;
  }
  t.stop(); t_init = t.elapsed_time();

  
  // output from GPU
  Mat    nborDistGPU(N*nLeaf, k);
  MatInt nborIDGPU(N*nLeaf, k);
  float *ptrDist[nLeaf];
  int   *ptrID[nLeaf];
  for (int i=0; i<nLeaf; i++) {
    ptrDist[i] = nborDistGPU.data()+i*N*k;
    ptrID[i]   = nborIDGPU.data()+i*N*k;
  }

  // GPU kernel
  float t_dist = 0., t_sort = 0., t_kernel = 0.;
  gemm_kselect_opt(nLeaf, R_ptr, Q_ptr, ID_ptr, N, d, ptrDist, ptrID, k, m,
      t_dist, t_sort, t_kernel);

  if (benchmark) {
    t_dist = 0., t_sort = 0., t_kernel = 0.;
    for (int i=0; i<repeat; i++) {
      gemm_kselect_opt(nLeaf, R_ptr, Q_ptr, ID_ptr, N, d, ptrDist, ptrID, k, m,
          t_dist, t_sort, t_kernel);
    }
    t_dist /= repeat;
    t_sort /= repeat;
    t_kernel /= repeat;
  }

  std::cout<<"Time for initialization: "<<t_init<<" s"<<std::endl;
  std::cout<<"Time for distance: "<<t_dist<<" s\n"
           <<"Time for sort: "<<t_sort<<" s\n"
           <<"Time for GEMM-kselect: "<<t_kernel<<" s\n\n";
  if (debug) {
    std::cout<<"GPU distance:\n"<<nborDistGPU<<std::endl
             <<"GPU ID:\n"<<nborIDGPU<<std::endl;
  }


  // CPU kernel
  if (nLeaf < 10) {
  Mat    nborDist(N*nLeaf, k);
  MatInt nborID(N*nLeaf, k);
  t.start();
  for (int i=0; i<nLeaf; i++) {
    float *dist = nborDist.data()+i*N*k;
    int   *ID   = nborID.data()+i*N*k;
    knn_cpu(R.block(i*N,0,N,d), Q.block(i*N,0,N,d), ID_ptr[i], N, d, dist, ID, k, debug);
  }
  t.stop();
  if (debug) {
    std::cout<<"CPU distance:\n"<<nborDist<<std::endl
             <<"CPU ID:\n"<<nborID<<std::endl;
  }
  std::cout<<"Check error of neighbor distance: "<<(nborDistGPU-nborDist).norm()/nborDist.norm()
           <<"\nCheck error of neighbor ID: "<<(nborIDGPU-nborID).norm()<<std::endl
           <<"CPU time: "<<t.elapsed_time()<<" s\n";

  if (false) {
  std::cout<<"Finding error ..."<<std::endl;
  for (int i=0; i<N*nLeaf; i++) {
    for (int j=0; j<k; j++) {
      if (nborIDGPU(i,j)!=nborID(i,j)) {
        std::cout<<"CPU ID: "<<nborID(i,j)<<", distance: "<<nborDist(i,j)<<std::endl
                 <<"GPU ID: "<<nborIDGPU(i,j)<<", distance: "<<nborDistGPU(i,j)<<std::endl;
      }
    }
  }
  }
  }

  return 0;
}


