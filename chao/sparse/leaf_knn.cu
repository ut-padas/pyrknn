#include "util_gpu.hpp"
#include "gemm.hpp"
#include "transpose.hpp"
#include "knn_handle.hpp"
#include "timer_gpu.hpp"
#include "matrix.hpp"

#include <string>
  

void find_neighbor(fvec&, int*, float*, int*, int, int, int, int, int, int, ivec&, float&, float&);

//void compute_row_norms(int n, int nnz, const ivec &rowPtrP, const fvec &valP, dvec<float> &P2);

void compute_row_norms(int n, int nnz, int *rowPtrP, float *valP, dvec<float> &P2) {
   
  iptr rowPtr(rowPtrP);
  dvec<int> rowIdx(nnz);
  thrust::counting_iterator<int> zero(0);
  thrust::upper_bound(rowPtr+1, rowPtr+n, zero, zero+nnz, rowIdx.begin());

  dvec<int> Prow(n);
  auto iterP2 = thrust::make_transform_iterator(fptr(valP), thrust::square<float>());
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

  
typedef dvec<int>::iterator ITER;
typedef thrust::permutation_iterator<dptr<int>, ITER> permT;

struct computeValIdx: public thrust::binary_function<int, int, int> {
  int *cumNnz;
  permT start;

  __host__ __device__
  computeValIdx(int *c, permT &s): cumNnz(c), start(s) {}

  __host__ __device__
  int operator()(int i, int blk) {
    return i-cumNnz[blk]+start[blk];
  }
};


struct computeRowIdx: public thrust::binary_function<int, int, int> {
  int *cumSize;
  int *start;

  __host__ __device__
  computeRowIdx(int *c, int *s): cumSize(c), start(s) {}

  __host__ __device__
  int operator()(int i, int blk) {
    return i-cumSize[blk]+start[blk];
  }
};


struct computeRowPtr: public thrust::binary_function<int, int, int> {
  permT start;
  int *cumNnz;

  __host__ __device__
  computeRowPtr(permT &s, int *c): start(s), cumNnz(c) {}

  __host__ __device__
  int operator()(int rowPtrP, int blk) {
    return rowPtrP - start[blk] + cumNnz[blk];
  }
};


void get_strided_block(int *rowPtrP, int *colIdxP, float *valP, 
    int maxPoint, int nNode, int b, int m,
    dvec<int> &Q_rowPtr, dvec<int> &Q_colIdx, dvec<float> &Q_val, 
    int &nRowQ, int &nnzQ) {
  
  ivec segments(nNode+1);
  thrust::sequence(segments.begin(), segments.end(), 0, maxPoint);
  int *seghead = thrust::raw_pointer_cast(segments.data());

  dvec<int> startRow(nNode), endRow(nNode);

  dptr<int> headPtr(seghead);
  thrust::constant_iterator<int> OFS1(b*m);
  thrust::constant_iterator<int> OFS2((b+1)*m);
  thrust::transform(headPtr, headPtr+nNode, OFS1, startRow.begin(), thrust::plus<int>()); 
  thrust::transform(headPtr, headPtr+nNode, OFS2, endRow.begin(), thrust::plus<int>()); 
  thrust::transform(headPtr+1, headPtr+nNode+1, endRow.begin(), endRow.begin(), thrust::minimum<int>());

  //tprint(startRow, "start row");
  //tprint(endRow, "end row");

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
  thrust::transform(zero, zero+nnzQ, blkIdx.begin(), valIdx.begin(), 
      computeValIdx(thrust::raw_pointer_cast(cumNnz.data()), startIdx));
  
  //dprint(nNode+1, cumNnz, "nnz");
  //dprint(nnzQ, valIdx, "value index");
  //dprint(nnzQ, blkIdx, "block index");
  
  auto copyVal = thrust::make_permutation_iterator(dptr<float>(valP), valIdx.begin());
  auto copyCol = thrust::make_permutation_iterator(dptr<int>(colIdxP), valIdx.begin());
  thrust::copy(copyVal, copyVal+nnzQ, Q_val.begin());
  thrust::copy(copyCol, copyCol+nnzQ, Q_colIdx.begin());


  // shift row pointers 
  dvec<int> rowSize(nNode);
  thrust::transform(endRow.begin(), endRow.end(), startRow.begin(), rowSize.begin(), thrust::minus<int>());
  nRowQ = thrust::reduce(rowSize.begin(), rowSize.end());
  dvec<int> cumRow(nNode+1, 0);
  thrust::inclusive_scan(rowSize.begin(), rowSize.end(), cumRow.begin()+1);

  blkIdx.resize(nRowQ);
  thrust::upper_bound(cumRow.begin()+1, cumRow.end(), zero, zero+nRowQ, blkIdx.begin());

  dvec<int> rowIdx(nRowQ);
  thrust::transform(zero, zero+nRowQ, blkIdx.begin(), rowIdx.begin(), 
      computeRowIdx(thrust::raw_pointer_cast(cumRow.data()), 
        thrust::raw_pointer_cast(startRow.data())));

  // compute row pointers
  Q_rowPtr.resize(nRowQ+1, nnzQ); // overwrite the first nRowQ entries
  auto copyRow = thrust::make_permutation_iterator(dptr<int>(rowPtrP), rowIdx.begin());
  thrust::transform(copyRow, copyRow+nRowQ, blkIdx.begin(), Q_rowPtr.begin(), 
      computeRowPtr(startIdx, thrust::raw_pointer_cast(cumNnz.data())));


  //tprint(Q_rowPtr, "rowPtrQ");
  //tprint(Q_colIdx, "colIdxQ");
  //tprint(Q_val, "valQ");
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
  
  dvec<int> denColIdx(nnzD);
  dptr<int> spColIdx(dD_colIdx);
  thrust::constant_iterator<int> N(maxPoint);
  thrust::transform(spColIdx, spColIdx+nnzD, N, denColIdx.begin(), thrust::modulus<int>());

  // treat as a csc format (notice the order of inputs are different from csr2dense)
  // output is a M[0]-by-b*nLeaf matrix in column-major
  int *colIdx = thrust::raw_pointer_cast(denColIdx.data());
  //dprint(m*nLeaf, maxPoint, nnzD, dD_rowPtr, den_colIdx, dD_val, "sparse");
  CHECK_CUSPARSE( cusparseScsc2dense(
        handle, maxPoint, m*nLeaf,
        descr, dD_val, colIdx, dD_rowPtr,
        Dist, maxPoint) )
  
  //dprint(nnzD, dD_val, "D value");
  //dprint(nnzD, den_colIdx, "D column index");
  //dprint(m*nLeaf+1, dD_rowPtr, "D row pointer");
  //dprint(m*nLeaf*maxPoint, Dist, "dist");
}


void compute_distance(dvec<int> &Q_rowPtr, dvec<int> &Q_colIdx, dvec<float> &Q_val, int nnzQ,
    int *rowPtrR, int *colIdxR, float *valR, int nnzR, dvec<float> &P2, dvec<float> &ones,
    int m, int nLeaf, int n, int d, int maxPoint, int offset,
    dvec<float> &D, float &t_sss, float &t_gemm, float &t_setup, float &t_den, float &sparse) {

  auto& handle = knnHandle_t::instance();

  // compute inner product
  int *rowPtrD, *colIdxD, nnzD;
  float *valD;
  
  int *rowPtrQ = thrust::raw_pointer_cast(Q_rowPtr.data());
  int *colIdxQ = thrust::raw_pointer_cast(Q_colIdx.data());
  float *valQ = thrust::raw_pointer_cast(Q_val.data());

  TimerGPU t; t.start();
  GEMM_SSS(m*nLeaf, n, d*nLeaf, -2, 
      rowPtrQ, colIdxQ, valQ, nnzQ,
      rowPtrR, colIdxR, valR, nnzR,
      rowPtrD, colIdxD, valD, nnzD,
      t_gemm, t_setup);
  sparse += 1.*nnzD/D.size();
  t.stop(); t_sss += t.elapsed_time();

  //dprint(m*nLeaf, n, nnzD, rowPtrD, colIdxD, valD, "GEMM");

  // dense format
  t.start();
  ivec segments(nLeaf+1);
  thrust::sequence(segments.begin(), segments.end(), 0, maxPoint);
  int *seghead = thrust::raw_pointer_cast(segments.data());
  float *Dist = thrust::raw_pointer_cast(D.data());
  form_dense_matrix(rowPtrD, colIdxD, valD, nnzD, 
      seghead, nLeaf, m, maxPoint, Dist, 
      handle.sparse, handle.mat);
  t.stop(); t_den += t.elapsed_time();
  
  //dprint(m*nLeaf, maxPoint, Dist, "dense");

  // rank-1 updates
  int N = maxPoint;
  int oneInt = 1; float oneFloat = 1.;
  float *ptrOne = thrust::raw_pointer_cast(ones.data());
  float *ptrP2 = thrust::raw_pointer_cast(P2.data());

  //t.start();
  CHECK_CUBLAS( cublasSgemmStridedBatched(
        handle.blas, CUBLAS_OP_N, CUBLAS_OP_T, 
        N, m, oneInt, &oneFloat, 
        ptrP2, N, N,
        ptrOne, m, m,
        &oneFloat, Dist, N, m*N, nLeaf) );

  CHECK_CUBLAS( cublasSgemmStridedBatched(
        handle.blas, CUBLAS_OP_N, CUBLAS_OP_T, 
        N, m, oneInt, &oneFloat, 
        ptrOne, N, N,
        ptrP2+offset, m, N,
        &oneFloat, Dist, N, m*N, nLeaf) );
  //t.stop(); t_rank += t.elapsed_time();
  
  // free memory from GEMM_SSS
  CHECK_CUDA( cudaFree(rowPtrD) )
  CHECK_CUDA( cudaFree(colIdxD) )
  CHECK_CUDA( cudaFree(valD) )
}


// ID[n]: gid of all points
// n: total number of points in all leaves
// m: blocking size
//void leaf_knn(ivec &ID, ivec &rowPtrP, ivec &colIdxP, fvec &valP, 
  //  int n, int d, int nnzP, int nLeaf, int maxPoint,
    //ivec &nborID, fvec &nborDist, int k, int m) {

void leaf_knn(int *ID, SpMat matrix, ivec &R_rowPtr, ivec &R_colIdx, fvec &R_val, int maxPoint,
    int *nborID, float *nborDist, int LD, int k, int m, fvec &Dist, ivec &tmp,
    float &t_trans, float &t_dist, float &t_sss, float &t_gemm, float &t_nnz, float &t_den, float &t_sort, float &sparse) {

  //dprint(n, d*nLeaf, nnzP, rowPtrP, colIdxP, valP, "Points");
  //dprint(n, ID, "ID");
  
  TimerGPU t;
  float t_nbor = 0., t_mat = 0., t_norm = 0.;
  float t_kcol = 0.;
  float t_row = 0.;

  int n = matrix.rows(), d = matrix.cols(), nnzP = matrix.nnz, nLeaf = matrix.nNodes; 
  int *rowPtrP = matrix.rowPtr, *colIdxP = matrix.colIdx;
  float *valP = matrix.val; 
  
  t.start();
  transpose(n, d*nLeaf, nnzP, rowPtrP, colIdxP, valP, R_rowPtr, R_colIdx, R_val, t_row);
  t.stop(); t_trans += t.elapsed_time();
  //std::cout<<"Finished transpose"<<std::endl;

  // point norm
  t.start();
  dvec<float> P2(n, 0.);
  compute_row_norms(n, nnzP, rowPtrP, valP, P2);
  t.stop(); t_norm += t.elapsed_time();

  // for rank-1 updates 
  dvec<float> ones(maxPoint*nLeaf, 1.);
  
  // loop over blocks
  int nBlock = (maxPoint + m-1)/m;
  for (int b=0; b<nBlock; b++) {
    // query points
    int nnzQ, nRowQ;
    dvec<int> Q_rowPtr, Q_colIdx;
    dvec<float> Q_val;
    t.start();
    get_strided_block(rowPtrP, colIdxP, valP, maxPoint, nLeaf, b, m,
        Q_rowPtr, Q_colIdx, Q_val, nRowQ, nnzQ);
    t.stop(); t_mat += t.elapsed_time();
    //dprint(nRowQ, d*nLeaf, nnzQ, Q_rowPtr, Q_colIdx, Q_val, "Query");


    // distance is a dense matrix
    t.start();
    int blockSize = std::min(m, maxPoint-b*m);
    assert(nRowQ == blockSize*nLeaf);
    int *rowPtrR = thrust::raw_pointer_cast(R_rowPtr.data());
    int *colIdxR = thrust::raw_pointer_cast(R_colIdx.data());
    float *valR  = thrust::raw_pointer_cast(R_val.data());

    compute_distance(Q_rowPtr, Q_colIdx, Q_val, nnzQ,
        rowPtrR, colIdxR, valR, nnzP, P2, ones, 
        blockSize, nLeaf, n, d, maxPoint, b*m, Dist, 
        t_sss, t_gemm, t_nnz, t_den, sparse);
    t.stop(); t_dist += t.elapsed_time();

    t.start();
    find_neighbor(Dist, ID, nborDist, nborID, nLeaf, blockSize, LD, k, maxPoint, b*m*LD, 
        tmp, t_sort, t_kcol);
    t.stop(); t_nbor += t.elapsed_time();
  }
}




