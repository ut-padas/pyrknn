#include <iostream>
#include <fstream>
#include <numeric> // std::iota
#include <string>
#include <vector>

#include <Eigen/Dense>
typedef Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> Mat;
typedef Eigen::Matrix<int,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> MatInt;
typedef Eigen::VectorXf Vec;
typedef Eigen::VectorXi VecInt;

#include "denknn.hpp"
#include "timer.hpp"
#include "readSVM.hpp"

void exact_knn(const Mat &Q, const Mat &R, const VecInt &ID, Mat &nborDist, MatInt &nborID);

void exact_knn(const Mat &Q, const Mat &R, const VecInt &ID, Mat &nborDist, MatInt &nborID, 
  const std::string &filename);


void write_matrix(const MatInt &nborID, const Mat &nborDist, const std::string &filename) {
  std::ofstream fout(filename.c_str());
  assert(fout.good());
  fout << nborID.rows() << " " << nborID.cols() << std::endl;
  fout << nborID << std::endl;
  fout << nborDist << std::endl;
  fout.close();
}


int compute_error(const MatInt &id, const Mat &dist, const MatInt &id_cpu, const Mat &dist_cpu, 
    int n, int k) {
  
  int miss = 0;
  for (int i=0; i<n; i++) {
    auto *start = id_cpu.data()+i*k;
    auto *end = id_cpu.data()+(i+1)*k;
    for (int j=0; j<k; j++) {
      if ( std::find(start, end, id(i,j)) == end )
        miss++;
    }
  }
  return miss;
}


Mat read_dataset(const std::string &dataset) {

  Mat P;
  Timer t; t.start();
  if (!dataset.compare("mnist"))
    P = read_mnist();
  else if (!dataset.compare("sphere")) {
    int m = 1<<21;
    int n = 200;
    std::vector<float> val(m*n);
    std::string filename("/scratch/06108/chaochen/will/sphere/sphere_set_0.bin");
    std::ifstream ifile(filename.c_str(), std::ios::in | std::ios::binary);
    ifile.read((char *)val.data(), m*n*sizeof(float));
    ifile.close();
    P = Eigen::Map<Mat>(val.data(), m, n);
  }
  t.stop();

  std::cout<<"\n======================\n"
           <<"Read data:\n"
           <<"----------------------\n"
           <<"dataset: "<<dataset<<"\n"
           <<"# rows: "<<P.rows()<<"\n"
           <<"# columns: "<<P.cols()<<"\n"
           <<"norm: "<<P.norm()<<"\n"
           <<"time: "<<t.elapsed_time()<<" s\n";
  std::cout<<"======================\n";
  return P;
}


Mat create_random_points(int argc, char *argv[]) {

  int n = 1024;
  int d = 64;
  int leaf = 1;
  for (int i=1; i<argc; i++) {
    if (!strcmp(argv[i],"-n"))
      n = atoi(argv[i+1]);
    if (!strcmp(argv[i],"-d"))
      d = atoi(argv[i+1]);
    if (!strcmp(argv[i],"-leaf"))
      leaf = atoi(argv[i+1]);
  }
  int N = n*leaf;
  assert(n>0);
  assert(d>0);

  // generate random sparse points
  Timer t; t.start();
  Mat P = Mat::Random(N, d)+Mat::Constant(N, d, 1.); // uniform [0,1]
  t.stop();
  std::cout<<"\nTime for generating points: "<<t.elapsed_time()<<" s\n";
  return P;
}


int main(int argc, char *argv[]) {
  
  Mat P;
  std::string dataset;
  int K = 5;
  int L = 3; // tree level
  int T = 1; // number of trees
  int blkTree = 10;
  int blkPoint = 64;
  for (int i=1; i<argc; i++) {
    if (!strcmp(argv[i],"-dataset"))
      dataset = std::string(argv[i+1]);
    if (!strcmp(argv[i],"-k"))
      K = atoi(argv[i+1]);
    if (!strcmp(argv[i],"-l"))
      L = atoi(argv[i+1]);
    if (!strcmp(argv[i],"-t"))
      T = atoi(argv[i+1]);
    if (!strcmp(argv[i],"-bt"))
      blkTree = atoi(argv[i+1]);
    if (!strcmp(argv[i],"-bp"))
      blkPoint = atoi(argv[i+1]);
  }
  if (dataset.empty()) { // create random data
    P = create_random_points(argc, argv);
  } else {
    P = read_dataset(dataset);
  }
  //std::cout<<"point on host:\n"<<P<<std::endl;
  int N = P.rows();
  int d = P.cols();
  VecInt ID = VecInt::LinSpaced(N, 0, N-1);
  assert(ID.size() == P.rows());

  std::cout.precision(2);
  std::cout<<std::scientific;
  std::cout<<"\n======================\n"
           <<"Inputs:\n"
           <<"----------------------\n"
           <<"N: "<<N<<std::endl
           <<"d: "<<d<<std::endl
           <<"k: "<<K<<std::endl
           <<"level: "<<L<<std::endl
           <<"trees: "<<T<<std::endl
           <<"----------------------\n"
           <<"block tree: "<<blkTree<<std::endl
           <<"block point: "<<blkPoint<<std::endl
           <<"======================\n\n";

  Timer t;
  float t_exact = 0., t_knn = 0.;

  t.start();
  int n = std::min(N, 100);
  Mat nborDistCPU(n, K);
  MatInt nborIDCPU(n, K);
  
  std::string filename;
  if (dataset.empty())
    exact_knn(P.topRows(n), P, ID, nborDistCPU, nborIDCPU);
  else {
    filename = "ref_"+dataset+".txt";
    exact_knn(P.topRows(n), P, ID, nborDistCPU, nborIDCPU, filename);
  }
  t.stop(); t_exact = t.elapsed_time();


  Mat nborDist = Mat::Constant(N, K, std::numeric_limits<float>::max());
  MatInt nborID = MatInt::Constant(N, K, std::numeric_limits<int>::max());
  int device = 0;
  int niter = (T+blkTree-1) / blkTree;
  std::vector<int> miss;
  for (int i=0; i<niter; i++) {
    t.start();
    int ntree = std::min(blkTree, T-i*blkTree);
    denknn(ID.data(), P.data(), N, d, L, ntree, nborID.data(), nborDist.data(), K, blkPoint, device);
    t.stop(); t_knn += t.elapsed_time();
    
    // compute error
    int err = compute_error(nborID, nborDist, nborIDCPU, nborDistCPU, n, K);
    std::cout<<"iter "<<i<<":\tmissed: "<<err<<"\t"<<"accuracy: "<<100.-1.*err/n/K*100<<" %\n";
    miss.push_back(err);
    if (1.*err/n/K < 0.05) break;
  }
  
  // output some results
  {
    int no = std::min(n, 10);
    int k = std::min(K, 5);

    std::cout<<"\t*** Results of first "<<no<<" points ***\n"
             <<"neighbor ID:\n"<<nborID.topLeftCorner(no, k)<<"\n"
             <<"neighbor distance:\n"<<nborDist.topLeftCorner(no, k)<<"\n"
             <<"Time for dense KNN: "<<t_knn<<" s\n"<<std::endl;

    std::cout<<"\n\t*** CPU reference ***\n"
             <<"neighbor ID:\n"<<nborIDCPU.topLeftCorner(no, k)<<"\n"
             <<"neighbor distance:\n"<<nborDistCPU.topLeftCorner(no, k)<<"\n"
             <<"CPU time: "<<t_exact<<" s\n";
  } 
  
  std::cout<<"\nError: \n";
  for (int i=0; i<(int)miss.size(); i++)
    printf("Trees %d, missed %d, accuracy: %.0f %%\n", i*blkTree, miss[i], 100.-1.*miss[i]/n/K*100);
  std::cout<<std::endl;

  if (!dataset.empty()) {
    filename[0]='g';
    filename[1]='p';
    filename[2]='u';
    write_matrix(nborID.topRows(n), nborDist.topRows(n), filename);
    std::cout<<"Finished computing results on GPU and writing them to "<<filename<<std::endl;
  }

  return 0;
}


Mat compute_distance(const Mat& R, const Mat& Q, bool debug=false) {
  Vec R2 = R.rowwise().squaredNorm();
  Vec Q2 = Q.rowwise().squaredNorm();
  Mat D2 = -2*Q*R.transpose();
  D2.colwise() += Q2;
  D2.rowwise() += R2.transpose();
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


void exact_knn(const Mat &Q, const Mat &R, const VecInt &ID, Mat &nborDist, MatInt &nborID) {
  Mat D = compute_distance(R, Q);
  int N = R.rows();
  int k = nborID.cols();
  for (int i=0; i<Q.rows(); i++) {
    kselect(D.data()+i*N, ID.data(), N, nborDist.data()+i*k, nborID.data()+i*k, k);
  }  
}

    
void exact_knn(const Mat &Q, const Mat &R, const VecInt &ID, Mat &nborDist, MatInt &nborID, 
  const std::string &filename) {

  // read precomputed results in file
  std::ifstream file(filename.c_str());
  if (file.good()) {
    int M, k;
    file >> M >> k;
    if (M >= Q.rows() && k >= nborID.cols()) {
      MatInt ID(M, k);
      Mat Dist(M, k);
      for (int i=0; i<M; i++)
        for (int j=0; j<k; j++)
          file >> ID(i, j);
      for (int i=0; i<M; i++)
        for (int j=0; j<k; j++)
          file >> Dist(i, j);
      file.close();
      std::cout<<"Finished reading precomputed results from "<<filename<<std::endl;
      nborID = ID.topLeftCorner(Q.rows(), nborID.cols());
      nborDist = Dist.topLeftCorner(Q.rows(), nborID.cols());
      return;
    }
    file.close();
  }
  
  exact_knn(Q, R, ID, nborDist, nborID);

  write_matrix(nborID, nborDist, filename);
  std::cout<<"Finished computing results on CPU and writing them to "<<filename<<std::endl;
}

