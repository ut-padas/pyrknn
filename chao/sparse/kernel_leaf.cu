#include "util_gpu.hpp"
#include "kernel_gemm.hpp"
#include "knn_handle.hpp"
#include "timer_gpu.hpp"

#include <string>
#include <thrust/for_each.h>
  

void transpose_gpu(int m, int n, int nnz, int *dP_rowPtr, int *dP_colIdx, float *dP_val,
    dvec<int> &dR_colPtr, dvec<int> &dR_rowIdx, dvec<float> &dR_val);

// the following four kernels are implemented in kernel_dist.cu
void get_transpose(int m, int n, int nnz, int *dP_rowPtr, int *dP_colIdx, float *dP_val,
    int *dR_colPtr, int *dR_rowIdx, float *dR_val, cusparseHandle_t &handle);

void compute_row_norms(int n, int nnz, int *rowPtrP, float *valP, dvec<float> &P2);

void get_strided_block(int *rowPtrP, int *colIdxP, float *valP, 
    int *seghead, int nNode, int b, int m,
    dvec<int> &Q_rowPtr, dvec<int> &Q_colIdx, dvec<float> &Q_val, 
    int &nRowQ, int &nnzQ);

void compute_distance(dvec<int> &Q_rowPtr, dvec<int> &Q_colIdx, dvec<float> &Q_val, int nnzQ,
    int *rowPtrR, int *colIdxR, float *valR, int nnzR, dvec<float> &P2, dvec<float> &ones,
    int m, int nLeaf, int n, int d, int *seghead, int maxPoint, int blkIdx,
    dvec<float> &D,  // output
    csrgemm2Info_t &info, cusparseHandle_t &spHandle, cusparseMatDescr_t &descr,
    cublasHandle_t &denHandle, float&, float&);


template <typename T>
void dprint(int n, dvec<T> x, const std::string &name) {
  dprint(n, thrust::raw_pointer_cast(x.data()), name);
}


struct printf_functor
{
  __host__ __device__
  void operator()(int x)
  {
    // note that using printf in a __device__ function requires
    // code compiled for a GPU with compute capability 2.0 or
    // higher (nvcc --arch=sm_20)
    printf("%d\n", x);
  }
};


struct firstKCols : public thrust::unary_function<int, int> {
  int k, LD;

  __host__ __device__
    firstKCols(int k_, int LD_): k(k_), LD(LD_)  {}

  __host__ __device__
    int operator()(int i) {
      return i/k*LD+i%k;
    }
};


struct findRowKCols: public thrust::unary_function<int, int> {
  int k;
  int m; // blocking size
  int N; // # points in every leaf
  int offset; // offset of this block
  int LD;
  const int *rowID;

  __host__ __device__
    findRowKCols(int k_, int m_, int N_, int o_, int LD_, const int *ID_): 
      k(k_), m(m_), N(N_), offset(o_), LD(LD_), rowID(ID_) {}

  __host__ __device__
    int operator()(int i) {
      return rowID[ i/(m*k)*N+i%(m*k)/k+offset ]*LD + i%k;
    }
};


// iLD: leading dimension of the input 
// oLD: LD of the output
void get_kcols_dist(const dvec<float> &D, float *Dk, const int *ID, 
    int nLeaf, int m, int k, int iLD, int offset, int oLD) {
  auto zero  = thrust::make_counting_iterator<int>(0);
  auto iterD = thrust::make_transform_iterator(zero, firstKCols(k, iLD));
  auto permD = thrust::make_permutation_iterator(D.begin(), iterD);
  auto iterK = thrust::make_transform_iterator(zero, findRowKCols(k, m, iLD, offset, oLD, ID));
  //std::cout<<"Iterator:"<<std::endl;
  //thrust::for_each(thrust::device, iterK, iterK+nLeaf*m*k, printf_functor());
  auto permK = thrust::make_permutation_iterator(thrust::device_ptr<float>(Dk), iterK);
  thrust::copy(permD, permD+nLeaf*m*k, permK);
}


struct firstKIdx: public thrust::unary_function<int, int> {
  int k;
  int m;
  int N; // # points in every leaf node
  const int *permIdx; // from 0 to m*iLD 

  __host__ __device__
    firstKIdx(int k_, int m_, int N_, const int *p_): 
      k(k_), m(m_), N(N_), permIdx(p_)  {}

  __host__ __device__
    int operator()(int i) {
      // i/k*N+i%k is the linear index for permIdx;
      // taking mod(N) is column index or local ID index.
      // i/(m*k)*N is the node index
      return permIdx[i/k*N+i%k]%N + i/(m*k)*N;
    }
};


void get_kcols_ID(const dvec<int> &permIdx, int *IDk, const int *ID,
    int nLeaf, int m, int k, int N, int offset, int oLD) {
  const int *pIdx  = thrust::raw_pointer_cast(permIdx.data());
  auto zero  = thrust::make_counting_iterator<int>(0);
  auto iterD = thrust::make_transform_iterator(zero, firstKIdx(k, m, N, pIdx));
  auto permD = thrust::make_permutation_iterator(thrust::device_ptr<const int>(ID), iterD);
  auto iterK = thrust::make_transform_iterator(zero, findRowKCols(k, m, N, offset, oLD, ID));
  auto permK = thrust::make_permutation_iterator(thrust::device_ptr<int>(IDk), iterK);
  thrust::copy(permD, permD+nLeaf*m*k, permK);
}


void find_neighbor(dvec<float> &Dist, int *ID, float *nborDist, int *nborID,
    int nLeaf, int m, int k, int N, int offset, int LD, mgpu::standard_context_t &ctx,
    dvec<int> &idx, float &t_sort, float &t_kcol) {
  
  // sorting
  dvec<int> segments(m*nLeaf);
  thrust::sequence(segments.begin(), segments.end(), 0, N);

  float *keys = thrust::raw_pointer_cast(Dist.data());
  int *vals = thrust::raw_pointer_cast(idx.data());
  int *segs = thrust::raw_pointer_cast(segments.data());
 
  //dprint(m*nLeaf, N, keys, "distance");
  //float t_sort, t_copy;
  TimerGPU t;
  t.start();
  mgpu::segmented_sort_indices(keys, vals, m*nLeaf*N, segs, m*nLeaf, mgpu::less_t<float>(), ctx);  
  t.stop(); t_sort += t.elapsed_time();
  
  //dprint(m*nLeaf, N, keys, "sorted distance");
  //dprint(m*nLeaf, N, vals, "index");
  

  // get first k
  t.start();
  get_kcols_dist(Dist, nborDist, ID, nLeaf, m, k, N, offset, LD);
  get_kcols_ID(idx, nborID, ID, nLeaf, m, k, N, offset, LD);
  t.stop(); t_kcol += t.elapsed_time();
}


// ID[n]: gid of all points
// n: total number of points in all leaves
// m: blocking size
void find_knn(int *ID, int *rowPtrP, int *colIdxP, float *valP, 
    int n, int d, int nnzP, int *seghead, int nLeaf, int m, int maxPoint,
    int *nborID, float *nborDist, int k, int LD) {

  //Access singleton knnHandle_t
  auto const& handle = knnHandle_t::instance();
    
  csrgemm2Info_t info = handle.info;
  cusparseHandle_t hCusparse = handle.hCusparse;
  cusparseMatDescr_t descr = handle.descr;
  cublasHandle_t hCublas = handle.hCublas; 
  mgpu::standard_context_t &ctx = *(handle.ctx);
  
  TimerGPU t;
  float t_mat = 0., t_dist = 0., t_nbor = 0., t_trans = 0.;
  float t_sort = 0., t_kcol = 0.;
  float t_gemm = 0., t_den = 0.;


  //dprint(n, d*nLeaf, nnzP, rowPtrP, colIdxP, valP, "Points");
  //dprint(n, ID, "ID");
  
  // R = P^T
  std::cout<<"[Transpose] R: "<<d/1.e9*nLeaf*4+nnzP/1.e9*4*2<<" GB"<<std::endl;
  dvec<int> R_rowPtr(d*nLeaf+1);
  dvec<int> R_colIdx(nnzP);
  dvec<float> R_val(nnzP);

  int *rowPtrR = thrust::raw_pointer_cast(R_rowPtr.data());
  int *colIdxR = thrust::raw_pointer_cast(R_colIdx.data());
  float *valR  = thrust::raw_pointer_cast(R_val.data());

  t.start();
#if 0
  get_transpose(n, d*nLeaf, nnzP, rowPtrP, colIdxP, valP,
      rowPtrR, colIdxR, valR, hCusparse);
#else
  transpose_gpu(n, d*nLeaf, nnzP, rowPtrP, colIdxP, valP,
      R_rowPtr, R_colIdx, R_val);
#endif
  t.stop(); t_trans = t.elapsed_time();
  std::cout<<"Finished transpose"<<std::endl;

  // point norm
  dvec<float> P2(n, 0.);
  compute_row_norms(n, nnzP, rowPtrP, valP, P2);

  // temporary array for distance calculation
  dvec<float> Dist(maxPoint*m*nLeaf);
  
  // for rank-1 updates 
  dvec<float> ones(maxPoint*nLeaf, 1.);
  
  // temporary array for sorting
  dvec<int> tmp(m*nLeaf*maxPoint); // no need to initialize for mgpu
  std::cout<<"Allocate distance and index array: "<<Dist.size()/1.e9*4*2<<" GB\n";

  // loop over blocks
  int nBlock = (maxPoint + m-1)/m;
  for (int b=0; b<nBlock; b++) {
    // query points
    int nnzQ, nRowQ;
    dvec<int> Q_rowPtr, Q_colIdx;
    dvec<float> Q_val;
    t.start();
    get_strided_block(rowPtrP, colIdxP, valP, seghead, nLeaf, b, m,
        Q_rowPtr, Q_colIdx, Q_val, nRowQ, nnzQ);
    t.stop(); t_mat += t.elapsed_time();
    //dprint(nRowQ, d*nLeaf, nnzQ, Q_rowPtr, Q_colIdx, Q_val, "Query");


    // distance is a dense matrix
    t.start();
    int blockSize = std::min(m, maxPoint-b*m);
    assert(nRowQ == blockSize*nLeaf);
    compute_distance(Q_rowPtr, Q_colIdx, Q_val, nnzQ,
        rowPtrR, colIdxR, valR, nnzP, P2, ones, 
        blockSize, nLeaf, n, d, seghead, maxPoint, b*m, Dist, 
        info, hCusparse, descr, hCublas, t_gemm, t_den);
    t.stop(); t_dist += t.elapsed_time();

    t.start();
    find_neighbor(Dist, ID, nborDist, nborID, nLeaf, blockSize, k, maxPoint, b*m, LD, ctx, 
        tmp, t_sort, t_kcol);
    t.stop(); t_nbor += t.elapsed_time();
  }

  std::cout<<"\n========================================"
           <<"\n\tKNN Timing"
           <<"\n----------------------------------------"
           <<"\n* Transpose: "<<t_trans<<" s"
           <<"\n* Get submatrix: "<<t_mat<<" s"
           <<"\n* Compute distance: "<<t_dist<<" s"
           <<"\n\t- gemm: "<<t_gemm<<" s"
           <<"\n\t- densify: "<<t_den<<" s"
           //<<"\n\t- rank-1: "<<t_rank<<" s"
           <<"\n* Find neighbors: "<<t_nbor<<" s"
           <<"\n\t- sort distance: "<<t_sort<<" s"
           <<"\n\t- get k-column: "<<t_kcol<<" s"
           <<"\n========================================"
           <<"\n"<<std::endl;
}




