#include <iostream>
#include <vector>
#include <numeric>

#include "merge_gpu.hpp"
#include "timer.hpp"

#include <Eigen/Dense>

namespace old {

using namespace Eigen;
typedef Matrix<float, Dynamic, Dynamic, RowMajor> Mat;
typedef Matrix<int, Dynamic, Dynamic, RowMajor> MatInt;

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

}

using namespace old;
int main(int argc, char *argv[]) {

  int m = 3;
  int n = 5;
  int k = 3;
  int repeat = 10;
  bool debug = false;
  bool benchmark = false;
  for (int i=1; i<argc; i++) {
    if (!strcmp(argv[i],"-n"))
      n = atoi(argv[i+1]);
    if (!strcmp(argv[i],"-m"))
      m = atoi(argv[i+1]);
    if (!strcmp(argv[i],"-k"))
      k = atoi(argv[i+1]);
    if (!strcmp(argv[i],"-D"))
      debug = true;
    if (!strcmp(argv[i],"-B"))
      benchmark = true;
  }
  assert(m > 0);
  assert(n >= k && k > 0);
  std::cout<<"\n======================\n"
           <<"Inputs:\n"
           <<"----------------------\n"
           <<"m: "<<m<<std::endl
           <<"n: "<<n<<std::endl
           <<"k: "<<k<<std::endl
           <<"repeat: "<<repeat<<std::endl
           <<"debug: "<<debug<<std::endl
           <<"benchmark: "<<benchmark<<std::endl
           <<"======================\n\n";

  Mat Dist;
  MatInt ID;

  float t_init;
  Timer t;
  
  // initialize
  t.start();
  //srand(11);
  Dist = Mat::Random(m, n);
  ID = MatInt(m, n);
  std::iota(ID.data(), ID.data()+m*n, 0);

  Mat D1(m, k), D2(m, k);
  MatInt I1(m, k), I2(m, k);
  std::vector<int> rid(n);
  std::iota(rid.begin(), rid.end(), 0);
  for (int i=0; i<m; i++) {
    std::random_shuffle(rid.begin(), rid.end());
    for (int j=0; j<k; j++) {
      int l = rid[j];
      D1(i,j) = Dist(i,l);
      I1(i,j) = ID(i,l);
    }
    std::random_shuffle(rid.begin(), rid.end());
    for (int j=0; j<k; j++) {
      int l = rid[j];
      D2(i,j) = Dist(i,l);
      I2(i,j) = ID(i,l);
    }
  }
  t.stop(); t_init = t.elapsed_time();
  std::cout<<"Initialization took: "<<t_init<<" s\n";


  Mat nborDist(m, k);
  MatInt nborID(m, k);

  float t_kernel = 0.;
  merge_neighbors(D1.data(), D2.data(), I1.data(), I2.data(), m, k, 
      nborDist.data(), nborID.data(), k, t_kernel, debug);

  if (benchmark) {
    t_kernel = 0.;
    for (int r=0; r<repeat; r++) {
      merge_neighbors(D1.data(), D2.data(), I1.data(), I2.data(), m, k, 
          nborDist.data(), nborID.data(), k, t_kernel);
    }
    t_kernel /= repeat;
  }
  std::cout<<"GPU merge took: "<<t_kernel<<" s\n";

  if (debug)
    std::cout<<"=== GPU ===\n"<<"neighbor ID:\n"<<nborID<<std::endl
           <<"neighbor Dist:\n"<<nborDist<<std::endl;

  
  float t_cpu;
  t.start();
  Mat nborDistCPU(m, k);
  MatInt nborIDCPU(m, k);
  for (int i=0; i<m; i++) {
    float *d1 = D1.data()+i*k;
    float *d2 = D2.data()+i*k;
    int *i1 = I1.data()+i*k;
    int *i2 = I2.data()+i*k;
    float *dist = nborDistCPU.data()+i*k;
    int *id = nborIDCPU.data()+i*k;
    merge_neighbor(d1, i1, k, d2, i2, k, dist, id, k);
  }
  t.stop(); t_cpu = t.elapsed_time();
  std::cout<<"CPU merge took: "<<t_cpu<<" s\n";
  
  if (debug)
    std::cout<<"=== CPU ===\n"<<"neighbor ID:\n"<<nborID<<std::endl
           <<"neighbor Dist:\n"<<nborDist<<std::endl;

  std::cout<<"Error of neighbor dist: "<<(nborDist-nborDistCPU).norm()<<std::endl
           <<"Error of neighbor ID: "<<(nborID-nborIDCPU).norm()<<std::endl;
  
  std::cout<<"Finding error ...\n";
  for (int i=0; i<m; i++) {
    for (int j=0; j<k; j++) {
      if (nborID(i,j) != nborIDCPU(i,j)) {

        std::cout<<"p["<<i<<"]\n";
        std::cout<<"-CPU- ";
        for (j=0; j<k; j++)
          std::cout<<"ID: "<<nborIDCPU(i,j)<<", distance: "<<nborDistCPU(i,j)<<" | ";
        std::cout<<std::endl;
        std::cout<<"-GPU- ";
        for (j=0; j<k; j++)
          std::cout<<"ID: "<<nborID(i,j)<<", distance: "<<nborDist(i,j)<<" | ";
        std::cout<<std::endl;
        
        break;

      }
    }
  }
  

  return 0;
}

