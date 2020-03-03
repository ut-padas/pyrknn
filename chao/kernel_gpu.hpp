#ifndef KERNEL_HPP
#define KERNEL_HPP

#define cudaCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
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

#define cublasCheck(ans) { cublasAssert((ans), __FILE__, __LINE__); }
inline void cublasAssert(cublasStatus_t code, const char *file, int line, bool abort=true) {
   if (code != CUBLAS_STATUS_SUCCESS) {
      fprintf(stderr,"CUBLAS assert: %s %s %d\n", cudaGetErrorEnum(code), file, line);
      if (abort) exit(code);
   }
}


template <typename T>
void print(const thrust::device_vector<T>& vec, const std::string &name) {
  std::cout<<std::endl<<name<<":"<<std::endl;
  for (int i=0; i<vec.size(); i++)
    std::cout<<vec[i]<<" ";
  std::cout<<std::endl<<std::endl;
}

template <typename T>
void print(const T *x, int N, const std::string &name) {
  std::cout<<std::endl<<name<<":"<<std::endl;
  for (int i=0; i<N; i++)
    std::cout<<x[i]<<" ";
  std::cout<<std::endl<<std::endl;
}

// note: functor inherits from unary_function
struct square : public thrust::unary_function<float,float>
{
  __host__ __device__
  float operator()(float x) const
  {
    return x*x;
  }
};

// convert a linear index to a row index
template <typename T>
struct linear_index_to_row_index : public thrust::unary_function<T,T>
{
    T C; // number of rows
    
    __host__ __device__
    linear_index_to_row_index(T _C) : C(_C) {}

    __host__ __device__
    T operator()(T i)
    {
        return i / C;
    }
};

template <typename T>
struct compare : public thrust::binary_function<int, int, bool> 
{
  const T *value;

  __host__ __device__
  compare(const T *value_) : value(value_) {}

  __host__ __device__
  bool operator()(int i, int j) {
    return value[i] < value[j];
  }
};

template <typename T>
struct equal : public thrust::binary_function<int, int, bool> 
{
  const T *value;

  __host__ __device__
  equal(const T *value_) : value(value_) {}

  __host__ __device__
  bool operator()(int i, int j) {
    return value[i] == value[j];
  }
};

template <typename T>
struct modID : public thrust::unary_function<int, T> {
  int k, N;
  const T *value;

  __host__ __device__
    modID(int k_, int N_, const T *value_) : k(k_), N(N_), value(value_)  {}

  __host__ __device__
    T operator()(int i) {
      int row = i/k;
      int col = i%k;
      return value[row*N+col];
    }
};


struct modDist : public thrust::unary_function<int, int> {
  int k, N;
  const int *value;

  __host__ __device__
    modDist(int k_, int N_, const int *value_) : k(k_), N(N_), value(value_)  {}

  __host__ __device__
    int operator()(int i) {
      int row = i/k;
      int col = i%k;
      return value[row*N+col]+row*N;
    }
};


#endif
