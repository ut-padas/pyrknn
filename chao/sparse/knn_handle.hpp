#include <moderngpu/kernel_segsort.hxx>
#include <memory>


template<typename T>
class Singleton {

  friend class knnHandle_t; // access private constructor/destructor

public:
  static T& instance() {
    static const std::unique_ptr<T> instance{new T()};
    return *instance;
  }

private:
  Singleton() {};
  ~Singleton() {};

  Singleton(const Singleton&) = delete;
  void operator=(const Singleton&) = delete;
};


class knnHandle_t final: public Singleton<knnHandle_t>{

  friend class Singleton<knnHandle_t>; // access private constructor/destructor

private:
  knnHandle_t() {
    std::cout<<"Create knnHandle_t instance"<<std::endl;
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

  mgpu::standard_context_t& mgpu_ctx() {return *ctx;}
 
public:
  ~knnHandle_t() {
    std::cout<<"Destroy knnHandle_t instance"<<std::endl;
    CHECK_CUSPARSE( cusparseDestroyCsrgemm2Info(info) )
    CHECK_CUSPARSE( cusparseDestroy(hCusparse) )
    CHECK_CUSPARSE( cusparseDestroyMatDescr(descr) )
    CHECK_CUBLAS( cublasDestroy(hCublas) )
    delete ctx;
  }

public:
  csrgemm2Info_t info;
  cusparseHandle_t hCusparse;
  cusparseMatDescr_t descr;
  cublasHandle_t hCublas; 
  mgpu::standard_context_t *ctx;
};



