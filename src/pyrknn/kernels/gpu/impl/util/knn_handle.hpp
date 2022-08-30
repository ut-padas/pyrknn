#ifndef KNN_HANDLE_HPP
#define KNN_HANDLE_HPP

#include "singleton.hpp"
#include <cusparse.h> 
#include "cublas_v2.h"

class knnHandle_t final: public Singleton<knnHandle_t>{
  friend class Singleton<knnHandle_t>; // access private constructor/destructor
private:
  knnHandle_t() {
    //std::cout<<"Create knnHandle_t instance"<<std::endl;
    // sparse info
    CHECK_CUSPARSE( cusparseCreateCsrgemm2Info(&info) )
    // sparse handle
    CHECK_CUSPARSE( cusparseCreate(&sparse) )
    // matrix descriptor
    CHECK_CUSPARSE( cusparseCreateMatDescr(&mat) )
    CHECK_CUSPARSE( cusparseSetMatType(mat, CUSPARSE_MATRIX_TYPE_GENERAL) )
    CHECK_CUSPARSE( cusparseSetMatIndexBase(mat, CUSPARSE_INDEX_BASE_ZERO) )
    // cublas handle
    CHECK_CUBLAS( cublasCreate(&blas) )
  }
public:
  ~knnHandle_t() {
    //std::cout<<"Destroy knnHandle_t instance"<<std::endl;
    CHECK_CUSPARSE( cusparseDestroyCsrgemm2Info(info) )
    CHECK_CUSPARSE( cusparseDestroy(sparse) )
    CHECK_CUSPARSE( cusparseDestroyMatDescr(mat) )
    CHECK_CUBLAS( cublasDestroy(blas) )
  }
public:
  csrgemm2Info_t info;
  cusparseHandle_t sparse;
  cusparseMatDescr_t mat;
  cublasHandle_t blas;
};

#endif
