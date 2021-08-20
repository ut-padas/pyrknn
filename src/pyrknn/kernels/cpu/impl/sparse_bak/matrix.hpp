#ifndef matrix_hpp
#define matrix_hpp

#include "util.hpp"
#include "omp.h"

#include <assert.h>
#include <algorithm>  // generate
#include <cstddef>    // NULL
#include <functional> // bind
#include <iostream>
#include <random>     // mt19937 and uniform_real_distribution
#include <vector>

typedef std::vector<unsigned> ivec;
typedef std::vector<float> fvec;
typedef std::vector<double> dvec;


class Points {
  public:
    Points(unsigned n_, unsigned d_, unsigned nnz_, int *rowPtr_, int *colIdx_, float *val_, 
        bool hasData_=false)
      : n(n_), d(d_), nnz(nnz_), rowPtr(rowPtr_), colIdx(colIdx_), val(val_), hasData(hasData_) 
    {}

    unsigned rows() const {return n;}
    
    unsigned cols() const {return d;}
    
    unsigned nonZeros() const {return nnz;}

    // return rows in [from, to)
    Points subset(unsigned from, unsigned to) const {
      assert(to <= n);
      unsigned n_ = to-from;
      unsigned d_ = d;
      unsigned nnz_ = rowPtr[to] - rowPtr[from];
      int *rowPtr_ = new int[n_+1];
      int *colIdx_ = new int[nnz_];
      float *val_  = new float[nnz_];
      bool hasData_ = true;
      std::copy(&colIdx[rowPtr[from]], &colIdx[rowPtr[to]], colIdx_);
      std::copy(&val[rowPtr[from]], &val[rowPtr[to]], val_);
      std::copy(&rowPtr[from], &rowPtr[to+1], rowPtr_);
      std::transform(rowPtr_, rowPtr_+n_+1, rowPtr_, std::bind2nd(std::minus<int>(), rowPtr[from]));
      return Points(n_,d_,nnz_,rowPtr_,colIdx_,val_,hasData_);
    }

    ~Points() {
      if (hasData) {
        assert(rowPtr!=NULL);
        assert(colIdx!=NULL);
        assert(val!=NULL);
        delete[] rowPtr, delete[] colIdx, delete[] val;
        rowPtr = colIdx = NULL, val = NULL;
      }
    }

  public:
    unsigned n, d, nnz;
    int *rowPtr, *colIdx;
    float *val;

    bool hasData = false;
};


// storage in column major 
template <typename T>
class matrix {
  public:
    matrix(size_t m_, size_t n_, T *val_)
      : m(m_), n(n_), val(val_) {}

    matrix(size_t m_, size_t n_, bool rowMajor_=false) 
      : m(m_), n(n_), rowMajor(rowMajor_) {
        assert(m > 0);
        assert(n > 0);
        assert(m*n > 0);
        val = new T[m*n];
        assert(val != NULL);
        hasData = true;
        //std::cout<<"[Malloc] m: "<<m<<", n: "<<n<<", m*n: "<<m*n<<std::endl;
      }

    size_t rows() const {return m;}
    
    size_t cols() const {return n;}

    T* data() {return val;}
      
    T* data() const {return val;}
      
    void rand() {
      int seed = current_time_nanoseconds();
#pragma omp parallel
      {
        size_t from, to; 
        par::compute_range(m*n, from, to);
        //std::default_random_engine eng(seed);
        std::minstd_rand eng(seed); eng.discard(from);
        //std::uniform_real_distribution<float> dist;
        std::normal_distribution<float> dist;
        generate(val+from, val+to, bind(dist, eng)); 
      }
    }

    T operator()(size_t i, size_t j) const {
      if (rowMajor)
        return val[j+i*n];
      else
        return val[i+j*m];
    }

    T& operator()(size_t i, size_t j) {
      if (rowMajor)
        return val[j+i*n];
      else
        return val[i+j*m];
    }

    ~matrix() {
      if (hasData) {
        assert(val != NULL);
        delete[] val;
        val = NULL;
        hasData = false;
        //std::cout<<"[Free] m: "<<m<<", n: "<<n<<", m*n: "<<m*n<<std::endl;
      }
    }
    
  public:
    size_t m, n;
    T *val;

    bool hasData = false;
    bool rowMajor = false;
};

typedef matrix<unsigned> iMatrix;
typedef matrix<float> fMatrix;


template <typename T>
void print(const matrix<T> &A, std::string name) {
  std::cout<<name<<" "<<A.rows()<<" x "<<A.cols()<<" :\n";
  for (unsigned i=0; i<A.rows(); i++) {
    for (unsigned j=0; j<A.cols(); j++)
      std::cout<<A(i,j)<<" ";
    std::cout<<std::endl;
  }
  std::cout<<std::endl;
}


#endif
