#include <iostream>
#include <random>

#include "exact.hpp"


void random_matrix(SpMat &A, int rseed) {
  float sparsity = 0.5;
  std::default_random_engine gen(rseed);
  std::uniform_real_distribution<float> dist(0.0,1.0);
  int M=A.rows(), N=A.cols();
  //float *val = new float[M*N];
  //init_random_gpu(val, M*N);
  std::vector<T> tripletList;
  for(int i=0; i<M; ++i) {
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


int num_reference(int nr, int rank) {
  return nr+rank;
}

int num_reference_all(int nr, int nproc) {
  int sum = 0;
  for (int i=0; i<nproc; i++) sum += num_reference(nr, i);
  return sum;
}

int random_seed(int rank) {
  return rank*13;
}

int main(int argc, char *argv[]) {

  int nproc, rank;

  CHECK_MPI( MPI_Init(&argc,&argv) );
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int d = 5;
  int k = 3;
  int nq = 3;
  int nr = 10;

  for (int i=1; i<argc; i++) {
    if (!strcmp(argv[i],"-d"))
      d = atoi(argv[i+1]);
    if (!strcmp(argv[i],"-k"))
      k = atoi(argv[i+1]);
    if (!strcmp(argv[i],"-q"))
      nq = atoi(argv[i+1]);
    if (!strcmp(argv[i],"-r"))
      nr = atoi(argv[i+1]);
  }
  
  if (rank == 0) {
    std::cout<<"\n======================\n"
             <<"Inputs:\n"
             <<"----------------------\n"
             <<"nq: "<<nq<<std::endl
             <<"nr: "<<nr<<std::endl
             <<"d: "<<d<<std::endl
             <<"k: "<<k<<std::endl
             <<"======================\n\n";
  }

  // random seed for random matrix
  int rseed;

  // query points
  SpMat Q(nq, d);

  rseed = nproc; // same seed on all procs
  random_matrix(Q, rseed);

  // reference points
  int nr_loc = num_reference(nr, rank);
  SpMat R(nr_loc, d);
  VecInt ID = VecInt::LinSpaced(nr_loc, rank*100, rank*100+nr_loc); // assume <100 points/rank

  rseed = random_seed(rank);
  random_matrix(R, rseed);
  
  // parallel KNN
  MatInt nborID(nq, k);
  Mat nborDist(nq, k);
  
  /*
  for (int i=0; i<nproc; i++) {
    if (i==rank) {
      std::cout<<"[Rank "<<rank<<"] Q:\n"<<Q<<std::endl<<"R:\n"<<R<<std::endl
        <<"ID:\n"<<ID.transpose()<<std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
  */
  
  exact_knn(
      Q.rows(), Q.cols(), Q.nonZeros(), Q.outerIndexPtr(), Q.innerIndexPtr(), Q.valuePtr(),
      R.rows(), R.cols(), R.nonZeros(), R.outerIndexPtr(), R.innerIndexPtr(), R.valuePtr(),
      ID.data(), k, nborID.data(), nborDist.data());

  /*
  for (int i=0; i<nproc; i++) {
    if (i==rank) {
      std::cout<<"[Rank "<<rank<<"]\n"
        <<"nborID:\n"<<nborID<<std::endl
        <<"nborDist::\n"<<nborDist<<std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
  */

  // reduce to rank 0
  merge(nq, k, nborID.data(), nborDist.data());
  
  /*
  for (int i=0; i<nproc; i++) {
    if (i==rank) {
      std::cout<<"[Rank "<<rank<<"]\n"
        <<"nborID:\n"<<nborID<<std::endl
        <<"nborDist::\n"<<nborDist<<std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
  */

  // serial KNN 
  if (rank==0) {
    int N = num_reference_all(nr, nproc);
    SpMat R0(N, d);
    VecInt ID0(N); 
    
    int row = 0;
    for (int i=0; i<nproc; i++) {
      int seed = random_seed(i);
      SpMat tmp(num_reference(nr, i), d);
      random_matrix(tmp, seed);
      R0.middleRows(row, tmp.rows()) = tmp;
    
      VecInt id = VecInt::LinSpaced(tmp.rows(), i*100, i*100+tmp.rows());
      ID0.segment(row, tmp.rows()) = id;
      
      row += tmp.rows();
    }
  

    MatInt nborIDRef(nq, k);
    Mat nborDistRef(nq, k);
    exact_knn(
        Q.rows(), Q.cols(), Q.nonZeros(), Q.outerIndexPtr(), Q.innerIndexPtr(), Q.valuePtr(),
        R0.rows(), R0.cols(), R0.nonZeros(), R0.outerIndexPtr(), R0.innerIndexPtr(), R0.valuePtr(),
        ID0.data(), k, nborIDRef.data(), nborDistRef.data());
    std::cout<<"[Rank "<<rank<<"] error: "<<(nborID-nborIDRef).norm()<<std::endl;
  }

  MPI_Finalize();
  return 0;
}

