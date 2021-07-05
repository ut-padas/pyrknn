#include <iostream>
#include <algorithm>

#include <Eigen/Dense>
using namespace Eigen;
typedef Matrix<float, Dynamic, Dynamic> Mat;

void sort_gpu(float*, int, int);

Mat sort_cpu(const Mat &A) {
  Mat B = A;
  int n = A.cols();
  float *ptr = B.data();
  for (int i=0; i<B.rows(); i++)
    std::sort(ptr+i*n, ptr+(i+1)*n);
  return B;
}

int main(int argc, char *argv[]) {
  
  int m = 5;
  int n = 7;
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
  Mat B = sort_cpu(A);
  sort_gpu(A.data(), m, n);

  std::cout<<"Error: "<<(B-A).norm()<<"\n";

  return 0;
}

