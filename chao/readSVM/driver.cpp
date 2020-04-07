#include <iostream>

#include "readSVM.hpp"
#include "timer.hpp"

int main(int argc, char *argv[]) {
  /*
  int nDays = 1;
  for (int i=0; i<argc; i++) {
    if (!strcmp(argv[i],"-days"))
      nDays = atoi(argv[i+1]);
  }
  std::cout<<"\nRead URL data: "<<nDays<<" days"<<std::endl;

  Timer t; t.start();
  SpMat A = read_url_dataset(nDays);
  t.stop();
  */

  Timer t; t.start();
  SpMat A = read_url_dataset(10);
  //SpMat A = read_avazu_dataset();
  //SpMat A = read_criteo_dataset();
  //SpMat A = read_kdd12_dataset();
  t.stop();

  std::cout<<"\n# rows: "<<A.rows()<<"\n"
           <<"# columns: "<<A.cols()<<"\n"
           <<"# nonzeros: "<<A.nonZeros()<<"\n"
           <<"sparsity: "<<100.*A.nonZeros()/A.rows()/A.cols()<<" %\n"
           <<"time: "<<t.elapsed_time()<<" s\n\n";

  //std::string filename("/scratch/06108/chaochen/criteo_csr.bin");
  //std::string filename("/scratch/06108/chaochen/kdd12_csr.bin");
  std::string filename("/scratch/06108/chaochen/url_10day_csr.bin");
  write_csr_binary(A, filename);
  
  t.start();
  SpMat B = read_csr_binary(filename);
  t.stop();

  std::cout<<"Error: "<<(A-B).norm()
    <<"\ntime: "<<t.elapsed_time()<<" s"<<std::endl;

  return 0;
}

