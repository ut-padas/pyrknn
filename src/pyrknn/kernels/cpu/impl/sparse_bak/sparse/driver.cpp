#include <iostream>
#include <fstream>
#include <random>
#include <vector>
#include <numeric> // std::iota
#include <string>
#include <cstdlib>

#include "spknn.hpp"
#include "timer.hpp"
#include "readSVM.hpp"

#include <Eigen/Sparse>
typedef Eigen::SparseMatrix<float,Eigen::RowMajor> SpMat; // row-major sparse matrix
typedef Eigen::Triplet<float> T;

#include <Eigen/Dense>
typedef Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> Mat;
typedef Eigen::Matrix<unsigned,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> MatInt;
typedef Eigen::VectorXf Vec;
typedef Eigen::Matrix<unsigned,Eigen::Dynamic,1> VecInt;

#include "util_eigen.hpp"


void exact_knn(const SpMat &Q, const SpMat &R, const VecInt &ID, Mat &nborDist, MatInt &nborID);

void exact_knn(const SpMat &Q, const SpMat &R, const VecInt &ID, Mat &nborDist, MatInt &nborID, 
    const std::string&);



SpMat read_dataset(std::string dataset) {

  SpMat P;
  Timer t; t.start();
  if (dataset.compare("url")==0)
    //P = read_csr_binary("/scratch/06108/chaochen/url_1day_csr.bin");
    //P = read_csr_binary("/scratch/06108/chaochen/url_10day_csr.bin");
    //P = read_csr_binary("/scratch/06108/chaochen/url_csr.bin");
    // frontera machine
    //P = read_csr_binary("/work/06108/chaochen/shared/url_10day_csr.bin");
    P = read_csr_binary("/work/06108/chaochen/shared/url_csr.bin");
  else if (dataset.compare("avazu")==0)
    //P = read_csr_binary("/scratch/06108/chaochen/avazu_csr.bin");
    P = read_csr_binary("/work/06108/chaochen/shared/avazu_csr.bin");
  else if (dataset.compare("criteo")==0)
    P = read_csr_binary("/work/06108/chaochen/shared/criteo_csr.bin");
  else if (dataset.compare("kdd")==0)
    P = read_csr_binary("/work/06108/chaochen/shared/kdd12_csr.bin");
  else if (dataset.compare("avazu_app")==0)
    P = read_csr_binary("/scratch/06108/chaochen/avazu_app_csr.bin");
  else if (dataset.compare("avazu_app_t")==0)
    P = read_csr_binary("/scratch/06108/chaochen/avazu_app_t_csr.bin");
  else if (dataset.compare("kdd")==0)
    P = read_csr_binary("/scratch/06108/chaochen/kdd12_csr.bin");
  else if (dataset.compare("criteo")==0)
    P = read_csr_binary("/scratch/06108/chaochen/criteo_csr.bin");
  else if (dataset.compare("test")==0) {
    int m = 524288; 
    int n = 10000;
    int nnz = 52428800;
   
    //std::string dir("/scratch/06108/chaochen/will/"); 
    std::string dir("/scratch1/06081/wlruys/shared/");
    std::string ptrFile = dir + "test_sparse_ptr.bin";
    std::string idxFile = dir + "test_sparse_idx.bin";
    std::string dataFile = dir + "test_sparse_data.bin";
      
    std::ifstream ifile(ptrFile.c_str(), std::ios::in | std::ios::binary);
    assert(ifile.good());
    
    std::vector<int> rowPtr(m+1);
    ifile.read((char *)rowPtr.data(), (m+1)*sizeof(int));
    ifile.close();
    std::vector<int> colIdx(nnz);
    ifile.open(idxFile.c_str(), std::ios::in | std::ios::binary);
    ifile.read((char *)colIdx.data(), nnz*sizeof(int));
    ifile.close();
    std::vector<float> val(nnz);
    ifile.open(dataFile.c_str(), std::ios::in | std::ios::binary);
    ifile.read((char *)val.data(), nnz*sizeof(float));
    ifile.close();
    P = Eigen::MappedSparseMatrix<float, Eigen::RowMajor>
                (m, n, nnz, rowPtr.data(), colIdx.data(), val.data());
  }
  else
    assert(false && "unknown dataset");
  t.stop();

  std::cout<<"\n======================\n"
           <<"Read data:\n"
           <<"----------------------\n"
           <<"dataset: "<<dataset<<"\n"
           <<"# rows: "<<P.rows()<<"\n"
           <<"# columns: "<<P.cols()<<"\n"
           <<"# nonzeros: "<<P.nonZeros()<<"\n"
           <<"time: "<<t.elapsed_time()<<" s\n";
  std::cout<<"======================\n";
 
  return P;
}


std::default_random_engine gen;
void init_random(SpMat &A, unsigned M, int N, float sparsity) {
  std::uniform_real_distribution<float> dist(0.0,1.0);
  //float *val = new float[M*N];
  //init_random_gpu(val, M*N);
  std::vector<T> tripletList;
  for(unsigned i=0; i<M; ++i) {
    for(int j=0; j<N; ++j) {
       //auto x = val[i*N+j];
       auto x = dist(gen);
       if (x < sparsity) {
           tripletList.push_back( T(i,j,x) );
       }
    }
  }
  A.setFromTriplets(tripletList.begin(), tripletList.end());
  A.makeCompressed();
  //delete[] val;
}


SpMat create_random_points(int argc, char *argv[]) {

  int n = 1024; // points per leaf node
  int d = 64;
  float sparsity = 0.5;
  for (int i=1; i<argc; i++) {
    if (!strcmp(argv[i],"-n"))
      n = atoi(argv[i+1]);
    if (!strcmp(argv[i],"-d"))
      d = atoi(argv[i+1]);
    if (!strcmp(argv[i],"-sparse"))
      sparsity = atof(argv[i+1]);
  }
  int N = n;
  assert(n>0);
  assert(d>0);
  assert(sparsity > 0.);

  // generate random sparse points
  SpMat P(N, d);
  P.reserve(Eigen::VectorXi::Constant(N, d*sparsity+3));

  float t0;
  Timer t; t.start();
  init_random(P, N, d, sparsity);
  t.stop(); t0 = t.elapsed_time();
  std::cout<<"\nTime for generating random points: "<<t0<<" s\n";
  
  return P;
}


struct almost_equal {
  float x;
  almost_equal(float x_): x(x_) {}
  bool operator()(float y) {
    return std::fabs(x-y) < 1e-4;
  }
};


int compute_error(const MatInt &id, const Mat &dist, const MatInt &id_cpu, Mat &dist_cpu, 
    int n, int k) {
  
  int miss = 0;
  for (int i=0; i<n; i++) {
    float *start = dist_cpu.data()+i*k;
    float *end = dist_cpu.data()+(i+1)*k;
    // dist may not be sorted
    std::vector<float> sdist(dist.data()+i*k, dist.data()+(i+1)*k);
    std::sort(sdist.begin(), sdist.end());
    for (int j=0; j<k; j++) {
      start = std::find_if(start, end, almost_equal(sdist[j]));
      if ( start != end ) {
        //std::cout<<"Found "<<j<<": "<<dist(i,j)<<" at "<<*start<<std::endl;
        start++;  
      } else { // not found
        //std::cout<<"\n[i="<<i<<"]: Gave up at "<<j<<", missed "<<k-j<<std::endl;
        miss += k-j;
        break; // no need to search for the rest
      }
    }
  }
  return miss;
}

int compute_error_bak(const MatInt &id, const Mat &dist, const MatInt &id_cpu, const Mat &dist_cpu, 
    int n, int k) {

  int miss = 0;
  for (int i=0; i<n; i++) {
    const unsigned int *start = id_cpu.data()+i*k;
    const float farthest = dist_cpu(i,k-1);
    for (int j=0; j<k; j++) {
      if (std::find(start, start+k, id(i,j)) == start+k // ID not found
          && dist(i,j) > farthest*(1+std::numeric_limits<float>::epsilon())
          ) 
      {
        miss++;
      }
    }
  }  
  return miss;
}


void write_matrix(const MatInt &nborID, const Mat &nborDist, const std::string &filename) {
  std::ofstream fout(filename.c_str());
  assert(fout.good());
  fout << nborID.rows() << " " << nborID.cols() << std::endl;
  fout << nborID << std::endl;
  fout << nborDist << std::endl;
  fout.close();
}


int main(int argc, char *argv[]) {
  
  SpMat P;
  std::string dataset;
  int K = 5;
  int L = 3; // tree level
  int T = 1; // number of trees
  int blkTree = 1;
  //int blkLeaf = 512;
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
    //if (!strcmp(argv[i],"-bl"))
      //blkLeaf = atoi(argv[i+1]);
    if (!strcmp(argv[i],"-bp"))
      blkPoint = atoi(argv[i+1]);
  }
  if (dataset.empty()) { // create random data
    P = create_random_points(argc, argv);
  } else {
    P = read_dataset(dataset);
  }
  //P = P.topRows(1<<26);

  unsigned N = P.rows();
  unsigned d = P.cols();
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
           //<<"block leaf: "<<blkLeaf<<std::endl
           <<"block point: "<<blkPoint<<std::endl
           <<"----------------------\n"
           <<"sizeof(# points): "<<sizeof(N)<<std::endl
           <<"sizeof(size_t): "<<sizeof(size_t)<<std::endl
           <<"sizeof(Eigen::Index): "<<sizeof(P.rows())<<std::endl
           <<"sizeof(StorageIndex) (signed): "<<sizeof(SpMat::StorageIndex)<<std::endl
           <<"======================\n\n";

  Timer t;
  float t_exact = 0., t_spknn = 0.;

  t.start();
  int n = std::min(N, unsigned(100));
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


  //std::cout<<"Points:\n"<<P<<std::endl;
  Mat nborDist = Mat::Constant(N, K, std::numeric_limits<float>::max());
  MatInt nborID = MatInt::Constant(N, K, std::numeric_limits<unsigned>::max());
  int cores = atoi(std::getenv("OMP_NUM_THREADS"));
  int niter = (T+blkTree-1) / blkTree;
  std::vector<int> miss;
  for (int i=0; i<niter; i++) {
    t.start();
    int ntree = std::min(blkTree, T-i*blkTree);
    spknn(ID.data(), P.outerIndexPtr(), P.innerIndexPtr(), P.valuePtr(), N, d, P.nonZeros(), 
        nborID.data(), nborDist.data(), K, L, ntree, blkPoint, cores);
    t.stop(); t_spknn += t.elapsed_time();
    //std::cout<<"Points:\n"<<P<<std::endl;
    
    // compute error
    int err = compute_error_bak(nborID, nborDist, nborIDCPU, nborDistCPU, n, K);
    double acc = 100. - 1.*err/n/K*100;
    //std::cout<<"iter "<<i<<":\tmissed: "<<err<<"\t"<<"accuracy: "<<acc<<" %\n";
    printf("iter %d:\tmissed: %d\taccuracy: %.2f %\n", i, err, acc);
    miss.push_back(err);
    if (acc > 95.) break;
  }
  
  // output some results
  {
    int no = std::min(n, 10);
    int k = std::min(K, 5);

    std::cout<<"\t*** Results of first "<<no<<" points ***\n"
             <<"neighbor ID:\n"<<nborID.topLeftCorner(no, k)<<"\n"
             <<"neighbor distance:\n"<<nborDist.topLeftCorner(no, k)<<"\n"
             <<"Time for sparse KNN: "<<t_spknn<<" s\n"<<std::endl;

    std::cout<<"\n\t*** CPU reference ***\n"
             <<"neighbor ID:\n"<<nborIDCPU.topLeftCorner(no, k)<<"\n"
             <<"neighbor distance:\n"<<nborDistCPU.topLeftCorner(no, k)<<"\n"
             <<"CPU time: "<<t_exact<<" s\n";
  } 
  
  std::cout<<"\nError: \n";
  for (int i=0; i<(int)miss.size(); i++)
    printf("Trees %d, missed %d, accuracy: %.1f %%\n", i*blkTree, miss[i], 100.-1.*miss[i]/n/K*100);
  std::cout<<std::endl;

  if (!dataset.empty()) {
    filename[0]='c';
    filename[1]='p';
    filename[2]='u';
    write_matrix(nborID.topRows(n), nborDist.topRows(n), filename);
    std::cout<<"Finished randomized KNN and writing them to "<<filename<<std::endl;
  }

  return 0;
}


Mat compute_distance(const SpMat& Q, const SpMat& R) {
  Vec R2 = rowNorm(R);
  Vec Q2 = rowNorm(Q);
  
  Timer t; t.start();
  Mat D2 = -2*Q*R.transpose();
  t.stop(); std::cout<<"[CPU exact] inner product: "<<t.elapsed_time()<<" s\n";
  
  D2.colwise() += Q2;
  D2.rowwise() += R2.transpose();
  //std::cout<<"Q norm:\n"<<Q2<<"\nR norm:\n"<<R2<<std::endl;
  
  return D2;
}

 
void kselect(const float *value, const unsigned *ID, unsigned n, float *kval, unsigned *kID, int k) {
  std::vector<unsigned> idx(n);
  std::iota(idx.begin(), idx.end(), 0);
  // find the large k items
  std::nth_element(idx.begin(), idx.begin()+k, idx.end(),
      [&value](unsigned i, unsigned j) {return value[i]<value[j];});
  // sort the largest k
  std::stable_sort(idx.begin(), idx.begin()+k,
      [&value](unsigned i, unsigned j) {return value[i]<value[j];});
  for (int i=0; i<k; i++) {
    int j = idx[i];
    kval[i] = value[j];
    kID[i] = ID[j];
  }
}

/*
void exact_knn(int* q_rowPtr, int* q_colIdx, float* q_val, unsigned int m, unsigned int q_nnz, int* r_rowPtr, int* q_colIdx, float* r_val, unsigned int n, unsigned int r_nnz, unsigned int d, int k, unsigned int *nborID, float* nborDist){



}
*/
/*
void exact_knn
(int nQ, int dQ, int nnzQ, int *rowPtrQ, int *colIdxQ, float *valQ,
 int nR, int dR, int nnzR, int *rowPtrR, int *colIdxR, float *valR,
 int *ID, int k, int *nborID, float *nborDist) {
  assert(dQ==dR);
  SpMat Q = Eigen::MappedSparseMatrix<float, Eigen::RowMajor>
              (nQ, dQ, nnzQ, rowPtrQ, colIdxQ, valQ);
  SpMat R = Eigen::MappedSparseMatrix<float, Eigen::RowMajor>
              (nR, dR, nnzR, rowPtrR, colIdxR, valR);
  // compute distance
  Vec Q2 = row_norm(Q);
  Vec R2 = row_norm(R);
  Mat D2 = -2*Q*R.transpose();    
  D2.colwise() += Q2;
  D2.rowwise() += R2.transpose();
  // find neighbor
  for (int i=0; i<nQ; i++) {
    kselect(D2.data()+i*nR, ID, nR, nborDist+i*k, nborID+i*k, k);
  }
}
*/

void exact_knn(const SpMat &Q, const SpMat &R, const VecInt &ID, Mat &nborDist, MatInt &nborID) {
  unsigned M = Q.rows();
  unsigned N = R.rows();
  int k = nborDist.cols();
  assert(Q.cols() == R.cols());
  assert(ID.size() == N);
  assert(nborID.cols() == k);

  //Timer t; t.start();
  Mat D = compute_distance(Q, R);
  //t.stop(); std::cout<<"[CPU exact] compute distance: "<<t.elapsed_time()<<" s\n";
  //std::cout<<"exact distance:\n"<<D<<std::endl;

  Timer t; t.start();
  for (unsigned i=0; i<M; i++) {
    kselect(&D(i,0), ID.data(), N, nborDist.data()+i*k, nborID.data()+i*k, k);
  }
  t.stop(); std::cout<<"[CPU exact] kselect: "<<t.elapsed_time()<<" s\n";
}


void exact_knn(const SpMat &Q, const SpMat &R, const VecInt &ID, Mat &nborDist, MatInt &nborID,
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



