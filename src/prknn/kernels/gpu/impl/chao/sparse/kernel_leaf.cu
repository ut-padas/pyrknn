#include "kernel_gpu.hpp"
#include "merge_gpu.hpp"

#include <string>
#include <moderngpu/kernel_segsort.hxx>
#include <thrust/for_each.h>
   
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


void get_transpose(int m, int n, int nnz, int *dP_rowPtr, int *dP_colIdx, float *dP_val,
    int *dR_colPtr, int *dR_rowIdx, float *dR_val, cusparseHandle_t &handle) {

  size_t bufferSize;
  CHECK_CUSPARSE( cusparseCsr2cscEx2_bufferSize(
        handle, m, n, nnz, dP_val, dP_rowPtr, dP_colIdx, 
        dR_val, dR_colPtr, dR_rowIdx, 
        CUDA_R_32F, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO,
        CUSPARSE_CSR2CSC_ALG1, &bufferSize) )
  
  void *buffer = NULL;
  CHECK_CUDA( cudaMalloc(&buffer, bufferSize) )
  
  CHECK_CUSPARSE( cusparseCsr2cscEx2(
        handle, m, n, nnz, dP_val, dP_rowPtr, dP_colIdx, 
        dR_val, dR_colPtr, dR_rowIdx, 
        CUDA_R_32F, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO,
        CUSPARSE_CSR2CSC_ALG1, buffer) )

  //if (debug) dprint(n, m, nnz, dR_colPtr, dR_rowIdx, dR_val, "Transpose");
  CHECK_CUDA( cudaFree(buffer) );
}


void compute_row_norms(int n, int nnz, int *rowPtrP, float *valP, dvec<float> &P2) {
   
  dvec<int> rowIdx(nnz);
  thrust::counting_iterator<int> zero(0);
  thrust::upper_bound(dptr<int>(rowPtrP)+1, dptr<int>(rowPtrP)+n, zero, zero+nnz, rowIdx.begin());

  dvec<int> Prow(n);
  dptr<float> ptrP(valP);
  auto iterP2 = thrust::make_transform_iterator(ptrP, thrust::square<float>());
  auto end = thrust::reduce_by_key(rowIdx.begin(), rowIdx.end(), iterP2, Prow.begin(), P2.begin());

  //dprint(n, thrust::raw_pointer_cast(Prow.data()), "P2 row indices");
  
  // handle zero rows
  dvec<float> P2_tmp;
  int s = end.first - Prow.begin();
  if (s < n) {
    P2_tmp.resize(n, 0.);
    auto perm = thrust::make_permutation_iterator(P2_tmp.begin(), Prow.begin());
    thrust::copy(P2.begin(), P2.begin()+s, perm);
    P2 = P2_tmp;
  }

  //dprint(n, thrust::raw_pointer_cast(P2.data()), "GPU point norm");
}


struct computeValIdx: public thrust::binary_function<int, int, int> {
  int *cumNnz;

  typedef dvec<int>::iterator ITER;
  typedef thrust::permutation_iterator<dptr<int>, ITER> permT;
  permT start;

  __host__ __device__
  computeValIdx(int *c, permT &s): cumNnz(c), start(s) {}

  __host__ __device__
  int operator()(int i, int blk) {
    return i-cumNnz[blk]+start[blk];
  }
};


struct computeRowIdx: public thrust::unary_function<int, int> {
  int m; // m rows for every leaf
  int *start;

  __host__ __device__
  computeRowIdx(int m_, int *s): m(m_), start(s) {}

  __host__ __device__
  int operator()(int i) {
    return i%m+start[i/m];
  }
};


struct computeRowPtr: public thrust::unary_function<int, int> {
  int m; // m rows for every leaf
  int *cumNnz;
  
  typedef dvec<int>::iterator ITER;
  typedef thrust::permutation_iterator<dptr<int>, ITER> permT;
  permT blockStart;

  __host__ __device__
  computeRowPtr(int m_, permT &s, int *c): m(m_), blockStart(s), cumNnz(c) {}

  __host__ __device__
  int operator()(int i, int rowStart) {
    return rowStart - blockStart[i/m] + cumNnz[i/m];
  }
};


void get_strided_block(int *rowPtrP, int *colIdxP, float *valP, 
    int *seghead, int nNode, int offset, int m,
    dvec<int> &Q_rowPtr, dvec<int> &Q_colIdx, dvec<float> &Q_val, int &nnzQ) {
  
  dvec<int> startRow(nNode), endRow(nNode);

  dptr<int> headPtr(seghead);
  thrust::constant_iterator<int> OFS1(offset);
  thrust::constant_iterator<int> OFS2(offset+m);
  thrust::transform(headPtr, headPtr+nNode, OFS1, startRow.begin(), thrust::plus<int>());
  thrust::transform(headPtr, headPtr+nNode, OFS2, endRow.begin(), thrust::plus<int>()); 

  // number of nonzeros in every block
  dvec<int> nnz(nNode);

  typedef dvec<int>::iterator ITER;
  thrust::permutation_iterator<dptr<int>, ITER> startIdx(dptr<int>(rowPtrP), startRow.begin());
  thrust::permutation_iterator<dptr<int>, ITER> endIdx(dptr<int>(rowPtrP), endRow.begin());
  thrust::transform(endIdx, endIdx+nNode, startIdx, nnz.begin(), thrust::minus<int>());
 
  // copy value and column index to Q
  nnzQ = thrust::reduce(nnz.begin(), nnz.end());
  Q_val.resize(nnzQ);
  Q_colIdx.resize(nnzQ);
  
  dvec<int> valIdx(nnzQ), blkIdx(nnzQ), cumNnz(nNode+1, 0);
  thrust::counting_iterator<int> zero(0);
  thrust::inclusive_scan(nnz.begin(), nnz.end(), cumNnz.begin()+1);
  thrust::upper_bound(cumNnz.begin()+1, cumNnz.end(), zero, zero+nnzQ, blkIdx.begin());
  // value index = local index + start 
  thrust::transform(zero, zero+nnzQ, blkIdx.begin(), valIdx.begin(), computeValIdx(
        thrust::raw_pointer_cast(cumNnz.data()),
        startIdx));
        //thrust::raw_pointer_cast(startRow.data())));
  
  //dprint(nNode+1, cumNnz, "nnz");
  //dprint(nnzQ, valIdx, "value index");
  //dprint(nnzQ, blkIdx, "block index");

  
  auto copyVal = thrust::make_permutation_iterator(dptr<float>(valP), valIdx.begin());
  auto copyCol = thrust::make_permutation_iterator(dptr<int>(colIdxP), valIdx.begin());
  thrust::copy(copyVal, copyVal+nnzQ, Q_val.begin());
  thrust::copy(copyCol, copyCol+nnzQ, Q_colIdx.begin());

  // shift row pointers 
  dvec<int> rowIdx(m*nNode);
  thrust::transform(zero, zero+m*nNode, rowIdx.begin(), computeRowIdx(
        m, thrust::raw_pointer_cast(startRow.data())));

  Q_rowPtr.resize(m*nNode+1, nnzQ); // overwrite the first m*nNode entries
  thrust::permutation_iterator<dptr<int>, ITER> copyRow(dptr<int>(rowPtrP), rowIdx.begin());
  thrust::transform(zero, zero+m*nNode, copyRow, Q_rowPtr.begin(), 
      computeRowPtr(m, startIdx, thrust::raw_pointer_cast(cumNnz.data())));


  //dprint(m*nNode+1, Q_rowPtr, "rowPtrQ");
  //dprint(nnzQ, Q_colIdx, "colIdxQ");
  //dprint(nnzQ, Q_val, "valQ");
}


struct localIdx: public thrust::binary_function<int, int, int> {
  int *seghead;

  __host__ __device__
  localIdx(int *s): seghead(s) {}

  __host__ __device__
  int operator()(int gid, int nodeId) {
    return gid - seghead[nodeId];
  }
};


// maxPoint: maximum number of points in a leaf node
void form_dense_matrix(int *dD_rowPtr, int *dD_colIdx, float *dD_val, int nnzD,
    int *seghead, int nLeaf, int m, int maxPoint, float *Dist, 
    cusparseHandle_t &handle, cusparseMatDescr_t &descr) {
  
  dvec<int> dense_colIdx(nnzD);
  dptr<int> sparse_colIdx(dD_colIdx);

  dvec<int> nodeIdx(nnzD);
  thrust::counting_iterator<int> zero(0);
  auto cumNnz = thrust::make_permutation_iterator(dptr<int>(dD_rowPtr), dptr<int>(seghead));
  thrust::upper_bound(cumNnz+1, cumNnz+nLeaf, zero, zero+nnzD, nodeIdx.begin());
  thrust::transform(sparse_colIdx, sparse_colIdx+nnzD, nodeIdx.begin(), dense_colIdx.begin(), 
      localIdx(seghead));

  //dprint(nLeaf+1, seghead, "segment head");
  //dprint(nnzD, nodeIdx, "node idx");
    
  // treat as a csc format (notice the order of inputs are different from csr2dense)
  // output is a M[0]-by-b*nLeaf matrix in column-major
  int *den_colIdx = thrust::raw_pointer_cast(dense_colIdx.data());
  CHECK_CUSPARSE( cusparseScsc2dense(
        handle, maxPoint, m*nLeaf,
        descr, dD_val, den_colIdx, dD_rowPtr,
        Dist, maxPoint) )
  
  //dprint(nnzD, dD_val, "D value");
  //dprint(nnzD, den_colIdx, "D column index");
  //dprint(m*nLeaf+1, dD_rowPtr, "D row pointer");
  
  //dprint(m*nLeaf*maxPoint, Dist, "dist");

  // TODO: padding for leaf nodes that don't have maxPoint 
}


void compute_distance(dvec<int> &Q_rowPtr, dvec<int> &Q_colIdx, dvec<float> &Q_val, int nnzQ,
    int *rowPtrR, int *colIdxR, float *valR, int nnzR, dvec<float> &P2,
    int m, int nLeaf, int n, int d, int *seghead, int maxPoint, int blkIdx,
    dvec<float> &D,  // output
    csrgemm2Info_t &info, cusparseHandle_t &spHandle, cusparseMatDescr_t &descr,
    cublasHandle_t &denHandle) {

    // compute inner product
    int *rowPtrD, *colIdxD, nnzD;
    float *valD;
    
    int *rowPtrQ = thrust::raw_pointer_cast(Q_rowPtr.data());
    int *colIdxQ = thrust::raw_pointer_cast(Q_colIdx.data());
    float *valQ  = thrust::raw_pointer_cast(Q_val.data());
    GEMM_SSS(m*nLeaf, n, d*nLeaf, -2, 
        rowPtrQ, colIdxQ, valQ, nnzQ,
        rowPtrR, colIdxR, valR, nnzR,
        rowPtrD, colIdxD, valD, nnzD,
        info, spHandle, descr);
  
    //dprint(m*nLeaf, n, nnzD, rowPtrD, colIdxD, valD, "GEMM");

    // dense format
    float *Dist = thrust::raw_pointer_cast(D.data());
    form_dense_matrix(rowPtrD, colIdxD, valD, nnzD, 
        seghead, nLeaf, m, maxPoint, Dist, 
        spHandle, descr);
    
    //dprint(m*nLeaf, maxPoint, Dist, "dense");

    // rank-1 updates
    int N = maxPoint;
    int oneInt = 1; float oneFloat = 1.;
    dvec<float> ones(maxPoint*nLeaf, 1.0); 
    float *ptrOne = thrust::raw_pointer_cast(ones.data());
    float *ptrP2 = thrust::raw_pointer_cast(P2.data());

    CHECK_CUBLAS( cublasSgemmStridedBatched(
          denHandle, CUBLAS_OP_N, CUBLAS_OP_T, 
          N, m, oneInt, &oneFloat, 
          ptrP2, N, N,
          ptrOne, m, m,
          &oneFloat, Dist, N, m*N, nLeaf) );

    CHECK_CUBLAS( cublasSgemmStridedBatched(
          denHandle, CUBLAS_OP_N, CUBLAS_OP_T, 
          N, m, oneInt, &oneFloat, 
          ptrOne, N, N,
          ptrP2+blkIdx*m, m, N,
          &oneFloat, Dist, N, m*N, nLeaf) );
}


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
  int b; // block index
  int LD;
  const int *rowID;

  __host__ __device__
    findRowKCols(int k_, int m_, int N_, int b_, int LD_, const int *ID_): 
      k(k_), m(m_), N(N_), b(b_), LD(LD_), rowID(ID_) {}

  __host__ __device__
    int operator()(int i) {
      return rowID[ i/(m*k)*N+b*m+i%(m*k)/k ]*LD + i%k;
    }
};


// iLD: leading dimension of the input 
// oLD: LD of the output
void get_kcols_dist(const dvec<float> &D, float *Dk, const int *ID, 
    int nLeaf, int m, int k, int iLD, int blkIdx, int oLD) {
  auto zero  = thrust::make_counting_iterator<int>(0);
  auto iterD = thrust::make_transform_iterator(zero, firstKCols(k, iLD));
  auto permD = thrust::make_permutation_iterator(D.begin(), iterD);
  auto iterK = thrust::make_transform_iterator(zero, findRowKCols(k, m, iLD, blkIdx, oLD, ID));
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
    int nLeaf, int m, int k, int N, int blkIdx, int oLD) {
  const int *pIdx  = thrust::raw_pointer_cast(permIdx.data());
  auto zero  = thrust::make_counting_iterator<int>(0);
  auto iterD = thrust::make_transform_iterator(zero, firstKIdx(k, m, N, pIdx));
  auto permD = thrust::make_permutation_iterator(thrust::device_ptr<const int>(ID), iterD);
  auto iterK = thrust::make_transform_iterator(zero, findRowKCols(k, m, N, blkIdx, oLD, ID));
  auto permK = thrust::make_permutation_iterator(thrust::device_ptr<int>(IDk), iterK);
  thrust::copy(permD, permD+nLeaf*m*k, permK);
}


void find_neighbor(dvec<float> &Dist, int *ID, float *nborDist, int *nborID,
    int nLeaf, int m, int k, int N, int blkIdx, int LD, mgpu::standard_context_t &ctx) {
  
  // sorting
  dvec<int> idx(m*nLeaf*N); // no need to initialize for mgpu
  dvec<int> segments(m*nLeaf);
  thrust::sequence(segments.begin(), segments.end(), 0, N);

  float *keys = thrust::raw_pointer_cast(Dist.data());
  int *vals = thrust::raw_pointer_cast(idx.data());
  int *segs = thrust::raw_pointer_cast(segments.data());
  
  //dprint(m*nLeaf, N, keys, "distance");
  
  mgpu::segmented_sort_indices(keys, vals, m*nLeaf*N, segs, m*nLeaf, mgpu::less_t<float>(), ctx);  

  //dprint(m*nLeaf, N, keys, "sorted distance");
  //dprint(m*nLeaf, N, vals, "index");
  

  // get first k
  get_kcols_dist(Dist, nborDist, ID, nLeaf, m, k, N, blkIdx, LD);
  get_kcols_ID(idx, nborID, ID, nLeaf, m, k, N, blkIdx, LD);
}


// ID[n]: gid of all points
// n: total number of points in all leaves
// m: blocking size
void find_knn(int *ID, int *rowPtrP, int *colIdxP, float *valP, 
    int n, int d, int nnzP, int *seghead, int nLeaf, int m, int maxPoint,
    int *nborID, float *nborDist, int k, int LD) {

  //Access singleton knnHandle_t
  knnHandle_t *handle;
  handle = handle->getInstance();

  csrgemm2Info_t info = handle->info;
  cusparseHandle_t hCusparse = handle->hCusparse;
  cusparseMatDescr_t descr = handle->descr;
  cublasHandle_t hCublas = handle->hCublas; 
  mgpu::standard_context_t &ctx = *(handle->ctx);

  //dprint(n, d*nLeaf, nnzP, rowPtrP, colIdxP, valP, "Points");
  //dprint(n, ID, "ID");
  
  // R = P^T
  dvec<int> R_rowPtr(d*nLeaf+1);
  dvec<int> R_colIdx(nnzP);
  dvec<float> R_val(nnzP);

  int *rowPtrR = thrust::raw_pointer_cast(R_rowPtr.data());
  int *colIdxR = thrust::raw_pointer_cast(R_colIdx.data());
  float *valR  = thrust::raw_pointer_cast(R_val.data());
  get_transpose(n, d*nLeaf, nnzP, rowPtrP, colIdxP, valP,
      rowPtrR, colIdxR, valR, hCusparse);

  // point norm
  dvec<float> P2(n, 0.);
  compute_row_norms(n, nnzP, rowPtrP, valP, P2);

  // loop over blocks
  assert(maxPoint%m == 0); // TODO
  int nBlock = (maxPoint + m-1)/m;
  for (int b=0; b<nBlock; b++) {
    // query points
    int nnzQ;
    dvec<int> Q_rowPtr, Q_colIdx;
    dvec<float> Q_val;
    get_strided_block(rowPtrP, colIdxP, valP, seghead, nLeaf, b*m, m,
        Q_rowPtr, Q_colIdx, Q_val, nnzQ);
  
    //dprint(m*nLeaf, d*nLeaf, nnzQ, Q_rowPtr, Q_colIdx, Q_val, "Query");

    // distance is a dense matrix
    dvec<float> Dist(maxPoint*m*nLeaf);
    compute_distance(Q_rowPtr, Q_colIdx, Q_val, nnzQ,
        rowPtrR, colIdxR, valR, nnzP, P2, 
        m, nLeaf, n, d, seghead, maxPoint, b, Dist, 
        info, hCusparse, descr, hCublas);

    find_neighbor(Dist, ID, nborDist, nborID, nLeaf, m, k, maxPoint, b, LD, ctx);
  }
}




