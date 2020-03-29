#ifndef KERNEL_GPU_SPARSE
#define KERNEL_GPU_SPARSE

#include "util.hpp"

#include <moderngpu/kernel_segsort.hxx>

#include <cuda_runtime.h>  // cudaMalloc, cudaMemcpy, etc.
#include <cusparse.h>      // cusparse
#include "cublas_v2.h"

#include <algorithm>
#include <numeric>
#include <stdio.h>         // printf
#include <stdlib.h>        // EXIT_FAILURE
#include <assert.h>
#include <cstring>


#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/binary_search.h>

template <typename T>
using dvec = thrust::device_vector<T>;
  
template <typename T>
using dptr = thrust::device_ptr<T>; 

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        assert(false);                                                         \
    }                                                                          \
}

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
        assert(false);                                                         \
    }                                                                          \
}

static const char *cudaGetErrorEnum(cublasStatus_t error) {
    switch (error) {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";

        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";

        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";

        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";

        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";

        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";

        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";

        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";
    }
    return "<unknown>";
}

#define CHECK_CUBLAS(ans) { cublasAssert((ans), __FILE__, __LINE__); }
inline void cublasAssert(cublasStatus_t code, const char *file, int line, bool abort=true) {
   if (code != CUBLAS_STATUS_SUCCESS) {
      fprintf(stderr,"CUBLAS assert: %s %s %d\n", cudaGetErrorEnum(code), file, line);
      if (abort) exit(code);
   }
}


template <typename T>
void print(const thrust::device_vector<T>& vec, const std::string &name) {
  std::cout<<name<<":"<<std::endl;
  for (int i=0; i<vec.size(); i++)
    std::cout<<vec[i]<<" ";
  std::cout<<std::endl<<std::endl;
}


void dprint(int m, int n, int nnz, dvec<int> &rowPtr, dvec<int> &colIdx, dvec<float> &val,
    const std::string &name);


void GEMM_SSD(int m, int n, int k, float alpha,
    int *csrRowPtrA, int *csrColIndA, float *csrValA, int nnzA,
    int *csrRowPtrB, int *csrColIndB, float *csrValB, int nnzB,
    int *csrRowPtrC, int *csrColIndC, float *csrValC, int nnzC,
    csrgemm2Info_t &info, cusparseHandle_t &handle, cusparseMatDescr_t &descr);


void GEMM_SSS(int m, int n, int k, float alpha,
    int *csrRowPtrA, int *csrColIndA, float *csrValA, int nnzA,
    int *csrRowPtrB, int *csrColIndB, float *csrValB, int nnzB,
    int* &csrRowPtrC, int* &csrColIndC, float* &csrValC, int &nnzC,
    csrgemm2Info_t &info, cusparseHandle_t &handle, cusparseMatDescr_t &descr);



template<typename T>
class Singleton {

    public:
        static T& getInstance() noexcept(std::is_nothrow_constructible<T>::value){
            if(instanceS.get() == 0){
                instanceS.reset( new T() );
            }
            return *instanceS;
        }

    private:
        Singleton() {};
        ~Singleton() {};

        Singleton(const Singleton&);
        void operator=(const Singleton&);

        static std::unique_ptr<T> instanceS;

};

class knnHandle_t : public Singleton<knnHandle_t>{

    friend class Singleton<knnHandle_t>;
      
    public:
      csrgemm2Info_t info;
      cusparseHandle_t hCusparse;
      cusparseMatDescr_t descr;
      cublasHandle_t hCublas; 
      mgpu::standard_context_t *ctx;

      mgpu::standard_context_t& mgpu_ctx() {return *ctx;}

    private:
      knnHandle_t() {

        // sparse info
        CHECK_CUSPARSE( cusparseCreateCsrgemm2Info(&info) )
        // sparse handle
        CHECK_CUSPARSE( cusparseCreate(&hCusparse) )
        // matrix descriptor
        CHECK_CUSPARSE( cusparseCreateMatDescr(&descr) )
        CHECK_CUSPARSE( cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL) )
        CHECK_CUSPARSE( cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO) )
        // cublas handle
        CHECK_CUBLAS( cublasCreate(&hCublas) )
        // mgpu context
        ctx = new mgpu::standard_context_t(false);

      }

      ~knnHandle_t() {
        CHECK_CUSPARSE( cusparseDestroyCsrgemm2Info(info) )
        CHECK_CUSPARSE( cusparseDestroy(hCusparse) )
        CHECK_CUSPARSE( cusparseDestroyMatDescr(descr) )
        CHECK_CUBLAS( cublasDestroy(hCublas) )
        delete ctx;
      }

};


#endif //HEADER

