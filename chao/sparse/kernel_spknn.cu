#include "kernel_gpu.hpp"
#include "merge_gpu.hpp"
#include "timer_gpu.hpp"

#include <thrust/random.h>
#include <limits>       // std::numeric_limits


// implemented in kernel_leaf.cu
void find_knn(int *ID, int *rowPtrP, int *colIdxP, float *valP, 
    int n, int d, int nnzP, int *seghead, int nLeaf, int m, int maxPoint,
    int *nborID, float *nborDist, int k, int LD, knnHandle_t *handle);


template <typename T>
T diff(T *x, T *y, int n) {
  dptr<T> xptr(x), yptr(y);
  dvec<T> err(n);
  thrust::transform(xptr, xptr+n, yptr, err.begin(), thrust::minus<T>());
  auto iter = thrust::make_transform_iterator(err.begin(), thrust::square<T>());
  return thrust::reduce(iter, iter+n);
}


void dprint(int m, int n, int nnz, dvec<int> &rowPtr, dvec<int> &colIdx, dvec<float> &val,
    const std::string &name) {
  dprint(m, n, nnz, 
      thrust::raw_pointer_cast(rowPtr.data()),
      thrust::raw_pointer_cast(colIdx.data()),
      thrust::raw_pointer_cast(val.data()),
      name);
}


struct prg: public thrust::unary_function<unsigned int, float> {
  float a, b;

  __host__ __device__
  prg(float _a=0.f, float _b=1.f) : a(_a), b(_b) {};

  __host__ __device__
  float operator()(const unsigned int n) const {
    thrust::default_random_engine rng;
    thrust::uniform_real_distribution<float> dist(a, b);
    rng.discard(n);
    return dist(rng);
  }
};


struct average: public thrust::binary_function<int, int, int> {

  __host__ __device__
  average() {}

  __host__ __device__
  int operator()(int a, int b) {
    return (a+b)/2;
  }
};


struct shiftColIdx: public thrust::binary_function<int, int, int> {
  int d;

  __host__ __device__
  shiftColIdx(int d_): d(d_) {}

  __host__ __device__
  int operator()(int idx, int node) {
    return idx%d + d*node;
  }
};


void permute_sparse_matrix(int m, int n, int nnzA, 
    int *rowPtrA, int *colIdxA, float *valA, dvec<int> &perm,
    int* &rowPtrB, int* &colIdxB, float* &valB, int &nnzB,
    csrgemm2Info_t &info, cusparseHandle_t &handle, cusparseMatDescr_t &descr) {
  
  // create sparse permutation matrix
  dvec<float> ones(m, 1.0);
  dvec<int> rowPtrP(m+1);
  thrust::sequence(rowPtrP.begin(), rowPtrP.end(), 0);

  int *P_rowPtr = thrust::raw_pointer_cast(rowPtrP.data());
  int *P_colIdx = thrust::raw_pointer_cast(perm.data());
  float *P_val  = thrust::raw_pointer_cast(ones.data());
  GEMM_SSS(m, n, m, 1.0,
      P_rowPtr, P_colIdx, P_val, m,
      rowPtrA, colIdxA, valA, nnzA,
      rowPtrB, colIdxB, valB, nnzB,
      info, handle, descr);
}


// input sparse matrix is modified inplace
// assume n = d * nSegment
// seghead: start position of next level
void create_matrix_next_level(int *rowPtrA, int *colIdxA, float *valA, int m, int n, int nnzA,
    dvec<int> &perm, int *seghead, int nSegment, int d,
    csrgemm2Info_t &info, cusparseHandle_t &handle, cusparseMatDescr_t &descr) {
  
  // compute B = Perm * A
  int *rowPtrB, *colIdxB, nnzB;
  float *valB;
  
  permute_sparse_matrix(m, n, nnzA, rowPtrA, colIdxA, valA,
      perm, rowPtrB, colIdxB, valB, nnzB,
      info, handle, descr);
  assert(nnzA == nnzB);

  // shift column indices
  dvec<int> shift(nnzB);
  thrust::counting_iterator<int> zero(0);
  auto cum_nnz = thrust::make_permutation_iterator(dptr<int>(rowPtrB), dptr<int>(seghead));
  thrust::upper_bound(cum_nnz+1, cum_nnz+nSegment+1, zero, zero+nnzB, shift.begin());

  //dprint(m+1, rowPtrB, "row pointer");
  //dprint(nSegment+1, seghead, "segment head");
  //print(shift, "node");

  dptr<int> dColIdx(colIdxB);
  thrust::transform(dColIdx, dColIdx+nnzB, shift.begin(), dColIdx, shiftColIdx(d));


  // overwrite results to input
  thrust::copy_n(thrust::device, dptr<int>(rowPtrB), m+1, dptr<int>(rowPtrA));
  thrust::copy_n(thrust::device, dptr<int>(colIdxB), nnzB, dptr<int>(colIdxA));
  thrust::copy_n(thrust::device, dptr<float>(valB), nnzB, dptr<float>(valA));

  // free allocation from calling GEMM_SSS
  CHECK_CUDA( cudaFree(rowPtrB) )
  CHECK_CUDA( cudaFree(colIdxB) )
  CHECK_CUDA( cudaFree(valB) )
}


// *** Input ***
// n = sum(N, nNode): total number of points
// N: number of points in every node
// valX[d*nNode]: assume random projections/vectors are given
// seghead[nNode+1]: start position of all segments/clusters
// segHeadNext: start position at next level
// *** Output ***
// median
// CSR format of the block sparse diagonal matrix
// permuted ID
void create_tree_next_level(int *ID, int *rowPtrP, int *colIdxP, float *valP, 
    int n, int d, int nnz, int *seghead, int *segHeadNext, int nNode,
    float *valX, float *median, knnHandle_t *handle) {


  csrgemm2Info_t info = handle->info;
  cusparseHandle_t hCusparse = handle->hCusparse;
  cusparseMatDescr_t descr = handle->descr;
  mgpu::standard_context_t &ctx = *(handle->ctx);

  // block diagonal for X
  dvec<int> rowPtrX(d*nNode+1);
  dvec<int> colIdxX(d*nNode);
  
  thrust::sequence(rowPtrX.begin(), rowPtrX.end(), 0); // one nonzero per row
  thrust::counting_iterator<int> zero(0);
  thrust::constant_iterator<int> DIM(d);
  thrust::transform(zero, zero+d*nNode, DIM, colIdxX.begin(), thrust::divides<int>());

  // block diagonal for Y = P * X
  dvec<int> rowPtrY(n+1);
  dvec<int> colIdxY(n);
  dvec<float> valY(n);
  thrust::sequence(rowPtrY.begin(), rowPtrY.end(), 0); // one nonzero per row

  // compute projections
  int *X_rowPtr = thrust::raw_pointer_cast(rowPtrX.data());
  int *X_colIdx = thrust::raw_pointer_cast(colIdxX.data());
  int *Y_rowPtr = thrust::raw_pointer_cast(rowPtrY.data());
  int *Y_colIdx = thrust::raw_pointer_cast(colIdxY.data());
  float *Y_val  = thrust::raw_pointer_cast(valY.data());

  //dprint(n, d*nNode, nnz, rowPtrP, colIdxP, valP, "P");
  //dprint(d*nNode, nNode, d*nNode, X_rowPtr, X_colIdx, valX, "X");

  GEMM_SSD(n, nNode, d*nNode, 1.0,
      rowPtrP, colIdxP, valP, nnz,
      X_rowPtr, X_colIdx, valX, d*nNode,
      Y_rowPtr, Y_colIdx, Y_val, n,
      info, hCusparse, descr);
  
  //print(valY, "projection");


  // sort 
  dvec<int> idx(n); //thrust::sequence(idx.begin(), idx.end(), 0);
  int *idxPtr = thrust::raw_pointer_cast(idx.data());
  mgpu::segmented_sort_indices(Y_val, idxPtr, n, seghead, nNode, mgpu::less_t<float>(), ctx);

  //print(idx, "index");

  // permute ID
  dvec<int> IDcpy(dptr<int>(ID), dptr<int>(ID)+n);
  auto permID = thrust::make_permutation_iterator(IDcpy.begin(), idx.begin());
  thrust::copy(thrust::device, permID, permID+n, ID);
  
  //dprint(n, ID, "permuted ID");
  
  // get median
  dvec<int> medpos(nNode); // index of median position
  dptr<int> segPtr(seghead);
  thrust::transform(segPtr, segPtr+nNode, segPtr+1, medpos.begin(), average());
  auto perm = thrust::make_permutation_iterator(valY.begin(), medpos.begin());
  thrust::copy(thrust::device, perm, perm+nNode, median);
  
  //dprint(nNode, median, "median");

  // create block diagonal matrix for next level
  create_matrix_next_level(rowPtrP, colIdxP, valP, n, d*nNode, nnz, 
      idx, segHeadNext, 2*nNode, d, 
      info, hCusparse, descr);
  
  //dprint(n, 2*d*nNode, nnz, rowPtrP, colIdxP, valP, "P next");
}


// m: blocking size in distance calculation
void spknn(int *hRowPtr, int *hColIdx, float *hVal, int n, int d, int nnz, int level,
    int *hNborID, float *hNborDist, int k, int m) {
  
  //print(n, d, nnz, hRowPtr, hColIdx, hVal, "host P");
  
  // copy data to GPU
  dvec<int> dRowPtr(hRowPtr, hRowPtr+n+1);
  dvec<int> dColIdx(hColIdx, hColIdx+nnz);
  dvec<float> dVal(hVal, hVal+nnz);
  

  //dprint(n, d, nnz, dRowPtr, dColIdx, dVal, "device P");

  // -----------------------
  //    Setup for SPKNN
  // -----------------------
  knnHandle_t *handle = new knnHandle_t();

  // compute number of points in every node
  dvec<int> nPoints[level];
  nPoints[0].resize(1, n); // root
  for (int i=1; i<level; i++) {
    nPoints[i].resize( 1<<i );
    // even index gets half from parent
    dvec<int> evenIdx( 1<<(i-1) );
    thrust::sequence(evenIdx.begin(), evenIdx.end(), 0, 2);
    auto TWO  = thrust::make_constant_iterator<int>(2);
    auto evenElm = thrust::make_permutation_iterator(nPoints[i].begin(), evenIdx.begin());
    thrust::transform(nPoints[i-1].begin(), nPoints[i-1].end(), TWO, evenElm, thrust::divides<int>());
    // odd index gets the rest half
    auto OddElm = thrust::make_permutation_iterator(nPoints[i].begin()+1, evenIdx.begin());
    thrust::transform(nPoints[i-1].begin(), nPoints[i-1].end(), evenElm, OddElm, thrust::minus<int>());
    //print(nPoints[i], "# points");
  }

  // compute segment starts for all levels
  dvec<int> seghead[level];
  for (int i=0; i<level; i++) {
    seghead[i].resize( (1<<i)+1, 0 );
    thrust::inclusive_scan(nPoints[i].begin(), nPoints[i].end(), seghead[i].begin()+1);
    //print(seghead[i], "segment heads");
  }

  // generate random arrays
  dvec<float> dX[level-1];
  for (int i=0; i<level-1; i++) {
    int nNode = 1<<i;
    dX[i].resize( nNode*d );
    auto start = thrust::make_counting_iterator<int>( (nNode-1)*d );
    thrust::transform(start, start+nNode*d, dX[i].begin(), prg());
    //print(dX[i], "X");
  }

  // output median in tree construction
  dvec<float> median[level-1];
  for (int i=0; i<level-1; i++)
    median[i].resize( 1<<i );
  

  // -----------------------
  // output
  // -----------------------
  dvec<int> dNborID(n*k, std::numeric_limits<int>::max());
  dvec<float> dNborDist(n*k, std::numeric_limits<float>::max());

  // -----------------------
  // timing
  // -----------------------
  float t_build = 0., t_knn = 0., t_merge = 0.;
  TimerGPU t;

  // -----------------------
  // Start SPKNN
  // -----------------------
  // build tree
  int nTree = 1;
  for (int tree=0; tree<nTree; tree++) {

    // create a copy of the input
    dvec<int> dRowPtrCpy = dRowPtr;
    dvec<int> dColIdxCpy = dColIdx;
    dvec<float> dValCpy  = dVal;
    int *d_rowPtr = thrust::raw_pointer_cast(dRowPtrCpy.data());
    int *d_colIdx = thrust::raw_pointer_cast(dColIdxCpy.data());
    float *d_val  = thrust::raw_pointer_cast(dValCpy.data());
    
    // create tree
    dvec<int> ID(n); // global ID of all points
    thrust::sequence(ID.begin(), ID.end(), 0);
    t.start();
    for (int i=0; i<level-1; i++) {
      create_tree_next_level(
          thrust::raw_pointer_cast(ID.data()),
          // CSR format of sparse data
          d_rowPtr, 
          d_colIdx,
          d_val,
          n, d, nnz, 
          thrust::raw_pointer_cast(seghead[i].data()), 
          thrust::raw_pointer_cast(seghead[i+1].data()), 
          1<<i, // # tree nodes
          thrust::raw_pointer_cast(dX[i].data()), // random projection
          thrust::raw_pointer_cast(median[i].data()), // output
          handle);
    }
    t.stop();
    t_build += t.elapsed_time();

#if 0
    // check ID
    // compute B = Perm * A
    int *rowPtrB, *colIdxB, nnzB;
    float *valB;
    
    int *rowPtrP = thrust::raw_pointer_cast(dRowPtr.data());
    int *colIdxP = thrust::raw_pointer_cast(dColIdx.data());
    float *valP  = thrust::raw_pointer_cast(dVal.data());
    permute_sparse_matrix(n, d, nnz, rowPtrP, colIdxP, valP,
        ID, rowPtrB, colIdxB, valB, nnzB,
        info, handle, descr);
    
    assert(nnz == nnzB);
    int errRowPtr = diff(rowPtrB, d_rowPtr, n+1);
    float errVal  = diff(valB, d_val, nnz);
    std::cout<<"Error of row pointer: "<<errRowPtr
             <<"\nError of value: "<<errVal
             <<std::endl;

    // free allocation from calling GEMM_SSS
    CHECK_CUDA( cudaFree(rowPtrB) )
    CHECK_CUDA( cudaFree(colIdxB) )
    CHECK_CUDA( cudaFree(valB) )
#endif
    

    // compute neighbors at leaf level
   
    dvec<int> curNborID(n*k, -1);
    dvec<float> curNborDist(n*k, -1.);
    int *curID = thrust::raw_pointer_cast(curNborID.data());
    float *curDist = thrust::raw_pointer_cast(curNborDist.data());
    

    int nLeaf = 1<<(level-1);
    int *leafPoints = thrust::raw_pointer_cast(nPoints[level-1].data());
    int maxPoint = thrust::reduce(thrust::device, leafPoints, leafPoints+nLeaf, 0, thrust::maximum<int>());
    int *segPtr = thrust::raw_pointer_cast(seghead[level-1].data());
    int offset = 0;
    int LD = k; // leading dimension
    // int LD = 2*k;
    //if (t==1) {offset = 0;}
    //else {offset = k;}
    int *preID = thrust::raw_pointer_cast(dNborID.data()+offset);
    float *preDist = thrust::raw_pointer_cast(dNborDist.data()+offset);

    t.start();
    find_knn(thrust::raw_pointer_cast(ID.data()), 
        d_rowPtr, d_colIdx, d_val,
        n, d, nnz, segPtr, nLeaf, m, maxPoint,
        curID, curDist, k, LD, 
        handle);
    t.stop();
    t_knn += t.elapsed_time();

    //dprint(n, k, curDist, "curDist");
    //dprint(n, k, curID, "curID");


    // update previous results
    t.start();
    merge_neighbors_gpu(preDist, preID, curDist, curID, n, k, k);
    t.stop();
    t_merge += t.elapsed_time();
  }
  

  std::cout<<"build tree: "<<t_build<<" s"
           <<"\nKNN: "<<t_knn<<" s"
           <<"\nmerge: "<<t_merge<<" s"
           <<std::endl;

  // -----------------------
  // Copy to CPU
  // -----------------------
  thrust::copy(dNborID.begin(), dNborID.end(), hNborID);
  thrust::copy(dNborDist.begin(), dNborDist.end(), hNborDist);
  
  // clean resource
  delete handle;
}

