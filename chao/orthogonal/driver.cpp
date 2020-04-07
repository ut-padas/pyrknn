#include <iostream>

#include <Eigen/Dense>
using namespace Eigen;
typedef Matrix<float, Dynamic, Dynamic> Mat;

void orthogonal_gpu(float*, int, int);

int main(int argc, char *argv[]) {
  
  int m = 5;
  int n = 3;
  for (int i=1; i<argc; i++) {
    if (!strcmp(argv[i],"-m"))
      m = atoi(argv[i+1]);
    if (!strcmp(argv[i],"-n"))
      n = atoi(argv[i+1]);
  }  

 std::cout<<"\n======================\n"
           <<"Inputs:\n"
           <<"----------------------\n"
           <<"m: "<<m<<std::endl
           <<"n: "<<n<<std::endl
           <<"======================\n\n";

  Mat A = Mat::Random(m, n);
  orthogonal_gpu(A.data(), m, n);

  Mat E = Mat::Identity(n, n) - A.transpose() * A;

  std::cout<<"Error: "<<E.norm()<<"\n";

  return 0;
}

