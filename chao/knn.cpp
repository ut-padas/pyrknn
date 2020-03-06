#include <iostream>

#include <vector>
#include <numeric> // std::iota
#include <algorithm> // std::stable_sort

#include <Eigen/Dense>

#include "knn.hpp"
#include "timer.hpp"

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


void merge_neighbor(const float *D2PtrL, const int *IDl, int kl, 
		    const float *D2PtrR, const int *IDr, int kr, 
		    float *nborDistPtr, int *nborIDPtr, int k) {
  std::vector<float> D2(kl+kr);
  std::vector<int> ID(kl+kr);
  std::memcpy(D2.data(), D2PtrL, sizeof(float)*kl);
  std::memcpy(D2.data()+kl, D2PtrR, sizeof(float)*kr);
  std::memcpy(ID.data(), IDl, sizeof(int)*kl);
  std::memcpy(ID.data()+kl, IDr, sizeof(int)*kr);
  // (sort and) unique
  std::vector<int> idx(kl+kr);
  std::iota(idx.begin(), idx.end(), 0);
  std::stable_sort(idx.begin(), idx.end(),
		   [&ID](int i, int j) {return ID[i] < ID[j];});
  std::stable_sort(ID.begin(), ID.end());
  std::vector<int> idx2(kl+kr);
  std::iota(idx2.begin(), idx2.end(), 0);
  std::unique(idx2.begin(), idx2.end(),
	                 [&ID](int i, int j) {return ID[i]==ID[j];});
  ID.erase(std::unique(ID.begin(), ID.end()), ID.end());
  std::vector<float> value(ID.size());
  for (size_t i=0; i<ID.size(); i++) {
    int j = idx2[i];
    value[i] = D2[ idx[j] ];
  }
  // call k-select
  kselect(value.data(), ID.data(), ID.size(), nborDistPtr, nborIDPtr, k);
}

int test_knn_vector(int argc, char* argv[]) {

  int n = 1024;
  int d = 64;
  int k = 64;
  int s = 1;
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
  std::cout<<"\n======================\n"
           <<"Inputs:\n"
           <<"----------------------\n"
           <<"N: "<<n<<std::endl
           <<"d: "<<d<<std::endl
           <<"k: "<<k<<std::endl
           <<"num_leaf: "<<s<<std::endl
           <<"repeat: "<<repeat<<std::endl
           <<"debug: "<<debug<<std::endl
           <<"benchmark: "<<benchmark<<std::endl
           <<"======================\n\n";

  const int nLeaf = s;
  std::vector<int> Nr(s,n);
  std::vector<int> Nq(s,n);
  assert(Nr.size() == (size_t)nLeaf);
  assert(Nq.size() == (size_t)nLeaf);
  std::vector<Mat> vecR;
  std::vector<Mat> vecQ;
  float *vecRptr[nLeaf], *vecQptr[nLeaf];
  for (int i=0; i<nLeaf; i++) {
    vecR.push_back( Mat::Random(Nr[i],d) );
    vecQ.push_back( Mat::Random(Nq[i],d) );
    vecRptr[i] = vecR[i].data();
    vecQptr[i] = vecQ[i].data();
  }

  std::vector<std::vector<int>> vecID(nLeaf);
  int *IDPtr[nLeaf];
  int firstID = 0;
  for (int j=0; j<nLeaf; j++) {
    vecID[j].resize(Nr[j]);
    std::iota(vecID[j].begin(), vecID[j].end(), firstID);
    firstID += Nr[j];
    IDPtr[j] = vecID[j].data();
  }


  std::vector<Mat> vecNborDist;
  std::vector<MatInt> vecNborID;
  
  if (!benchmark) { // skip (slow) computation on CPU
  double t_cpu = 0.;
  Timer t; t.start();
  std::vector<Mat> vecD2;
  for (int i=0; i<nLeaf; i++) {
    vecD2.push_back( distSquared_cpu(vecR[i],vecQ[i], debug) );
  }
 
  if (debug) {
    print(vecD2, "D2");
  }


  /*
   * Compute k-nearest neighbors
   */
  for (int j=0; j<nLeaf; j++) {
    vecNborDist.push_back( Mat(Nq[j],k) );
    vecNborID.push_back( MatInt(Nq[j],k) );
    for (int i=0; i<Nq[j]; i++) {
      float *D2Ptr = vecD2[j].data() + i*Nr[j];
      float *nborDistPtr = vecNborDist[j].data() + i*k;
      int   *nborIDPtr = vecNborID[j].data() + i*k;
      kselect(D2Ptr, IDPtr[j], Nr[j], nborDistPtr, nborIDPtr, k);
    }   
  }
  t.stop(); t_cpu += t.elapsed_time();
  std::cout<<"Computation on CPU: "<<t_cpu<<" s."<<std::endl;
  }


  /*
   * Blocked batched GEMM fused with k-select
   */
 
  std::vector<Mat> vecNborDistBBF;
  std::vector<MatInt> vecNborIDBBF;
  float *ptrDist[nLeaf];
  int   *ptrID[nLeaf];
  for (int i=0; i<nLeaf; i++) {
    vecNborDistBBF.push_back( Mat::Zero(Nq[i],k) );
    vecNborIDBBF.push_back( MatInt::Zero(Nq[i],k) );
    ptrDist[i] = vecNborDistBBF[i].data();
    ptrID[i] = vecNborIDBBF[i].data();
  }

  // warm up the GPU
  float t_dist = 0., t_sort = 0., t_store = 0., t_kernel = 0.;
  bb_gemm_kselect(nLeaf, vecRptr, vecQptr, IDPtr, n, d, ptrDist, ptrID, k,
      t_dist, t_sort, t_store, t_kernel);

  t_dist = 0., t_sort = 0., t_store = 0., t_kernel = 0.;
  for (int i=0; i<repeat; i++) {
    bb_gemm_kselect(nLeaf, vecRptr, vecQptr, IDPtr, n, d, ptrDist, ptrID, k,
        t_dist, t_sort, t_store, t_kernel);
  }
  std::cout<<"Time for distance: "<<t_dist/repeat<<" s\n"
           <<"Time for sort: "<<t_sort/repeat<<" s\n"
           <<"Time for store: "<<t_store/repeat<<" s\n"
           <<"Time for GEMM-kselect: "<<t_kernel/repeat<<" s\n\n";
          
 
  if (!benchmark) {
  double NrmNborDist = 0., ErrNborDist = 0., ErrNborID = 0.;
  for (int i=0; i<nLeaf; i++) {
    NrmNborDist += vecNborDist[i].norm();
    ErrNborDist += (vecNborDist[i]-vecNborDistBBF[i]).norm();
    ErrNborID += (vecNborID[i]-vecNborIDBBF[i]).norm();
  }
  std::cout<<"Check (relative) error of neighbor distance: "
           <<ErrNborDist/NrmNborDist<<std::endl
	         <<"Check error of neighbor ID: "<<ErrNborID<<std::endl;

  if (false) { 
  std::cout<<"Finding error ..."<<std::endl;
  for (int i=0; i<nLeaf; i++) {
    for (int r=0; r<n; r++) {
      for (int c=0; c<k; c++) {
      if (vecNborID[i](r,c) != vecNborIDBBF[i](r,c)) {
        std::cout<<"true ID: "<<vecNborID[i](r,c)
                 <<", true distance: "<<vecNborDist[i](r,c)<<std::endl
                 <<"gpu ID: "<<vecNborIDBBF[i](r,c)
                 <<", gpu distance: "<<vecNborDistBBF[i](r,c)<<std::endl;
      }
      }
    }
  }
  }
  }
  

  return 0;
}

void test_single_leaf_node() {

  int Nr = 5;
  int Nq = 7;
  int d = 3;
  Mat R = Mat::Random(Nr, d);
  Mat Q = Mat::Random(Nq, d);
  
  //std::cout<<"R:\n"<<R<<std::endl
  //	   <<"Q:\n"<<Q<<std::endl;
  
  Mat D2 = distSquared_cpu(R, Q);
  Mat D2t(Nr,Nq);
  distSquared_gpu(R.data(), Q.data(), D2t.data(), R.rows(), Q.rows(), d);

  //std::cout<<"D2_cpu:\n"<<D2_cpu<<std::endl;
  //std::cout<<"D2_gpu:\n"<<D2_gpu<<std::endl;

  Mat ErrD2 = D2 - D2t.transpose();
  std::cout<<"Check error of distance: "<<ErrD2.norm()<<std::endl;

  /*
   * Compute K-nearest neighbors
   */

  int k = 3;
  assert(k < Nr);
  Mat nborDist(Nq, k);
  Mat nborDistGPU(Nq, k);
  MatInt nborID(Nq, k);
  MatInt nborIDGPU(Nq, k);
  std::vector<int> ID(Nr);
  std::iota(ID.begin(), ID.end(), 0);
  for (int i=0; i<Nq; i++) {
    float *D2Ptr = D2.data() + i*Nr;
    float *nborDistPtr = nborDist.data() + i*k;
    int *nborIDPtr = nborID.data() + i*k;
    kselect(D2Ptr, ID.data(), Nr, nborDistPtr, nborIDPtr, k);
    float *nborDistPtrGPU = nborDistGPU.data() + i*k;
    int *nborIDPtrGPU = nborIDGPU.data() + i*k;
    kselect_gpu(D2Ptr, ID.data(), Nr, nborDistPtrGPU, nborIDPtrGPU, k);
  } 

  /*
  std::cout<<"CPU"<<std::endl
	   <<"Neighbor Distance: \n"<<nborDist<<std::endl
	   <<"Neighbor ID:\n"<<nborID<<std::endl;
  
  std::cout<<"GPU"<<std::endl
	   <<"Neighbor Distance: \n"<<nborDistGPU<<std::endl
	   <<"Neighbor ID:\n"<<nborIDGPU<<std::endl;
  */

  Mat ErrNborDist = nborDist - nborDistGPU;
  MatInt ErrNborID = nborID - nborIDGPU;
  std::cout<<"Check error of neighbor distance: "<<ErrNborDist.norm()<<std::endl
	   <<"Check error of neighbor ID: "<<ErrNborID.norm()<<std::endl;

  /*
   * Merge two neighbor lists
   */

  int kl = Nr - 1;
  int kr = Nr - 1;
  Mat Rl = R.topRows(kl);
  Mat Rr = R.bottomRows(kr);
  Mat D2l = distSquared_cpu(Rl, Q);
  Mat D2r = distSquared_cpu(Rr, Q);
  Mat nborDistMg(Nq, k);
  Mat nborDistMgGPU(Nq, k);
  MatInt nborIDMg(Nq, k);
  MatInt nborIDMgGPU(Nq, k);
  std::vector<int> IDl(kl);
  std::vector<int> IDr(kr);
  std::iota(IDl.begin(), IDl.end(), 0);
  std::iota(IDr.begin(), IDr.end(), Nr-kr);
  for (int i=0; i<Nq; i++) {
    float *D2PtrL = D2l.data() + i*kl;
    float *D2PtrR = D2r.data() + i*kr;
    float *nborDistPtr = nborDistMg.data() + i*k;
    int *nborIDPtr = nborIDMg.data() + i*k;
    merge_neighbor(D2PtrL, IDl.data(), kl, D2PtrR, IDr.data(), kr, nborDistPtr, nborIDPtr, k); 
    float *nborDistPtrGPU = nborDistMgGPU.data() + i*k;
    int *nborIDPtrGPU = nborIDMgGPU.data() + i*k;
    merge_neighbor_gpu(D2PtrL, IDl.data(), kl, D2PtrR, IDr.data(), kr, nborDistPtrGPU, nborIDPtrGPU, k);
  }

  Mat ErrNborDistMg = nborDist - nborDistMg;
  MatInt ErrNborIDMg = nborID - nborIDMg;
  std::cout<<"Check error of merged distance: "<<ErrNborDistMg.norm()<<std::endl
	   <<"Check error of merged ID: "<<ErrNborIDMg.norm()<<std::endl;
  Mat ErrNborDistMgGPU = nborDistMgGPU - nborDistMg;
  MatInt ErrNborIDMgGPU = nborIDMgGPU - nborIDMg;
  std::cout<<"Check error of GPU merged distance: "<<ErrNborDistMgGPU.norm()<<std::endl
	   <<"Check error of GPU merged ID: "<<ErrNborIDMgGPU.norm()<<std::endl;

  /*
  std::cout<<"CPU"<<std::endl
	   <<"Neighbor Distance: \n"<<nborDistMg<<std::endl
	   <<"Neighbor ID:\n"<<nborIDMg<<std::endl;
  std::cout<<"GPU"<<std::endl
	   <<"Neighbor Distance: \n"<<nborDistMgGPU<<std::endl
	   <<"Neighbor ID:\n"<<nborIDMgGPU<<std::endl;
  */
}

