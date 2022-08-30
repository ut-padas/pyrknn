#include "NLA.hpp"
#include "mkl.h"
#include "timer.hpp"

#include <algorithm>

#define CHECK(func)                                                            \
{                                                                              \
    int info  = (func);                                                        \
    if (info != 0) {                                                           \
        printf("MKL failed at line %d: the %d-th parameter is illegal\n",      \
               __LINE__, -info);                                               \
        assert(false);                                                         \
    }                                                                          \
}

#define CHECKS(func)                                                           \
{                                                                              \
    sparse_status_t stat  = (func);                                                        \
    if (stat != SPARSE_STATUS_SUCCESS) {                                       \
        printf("MKL sparse failed at line %d: error %s\n",                     \
               __LINE__, GetErrorString(stat));                                \
        assert(false);                                                         \
    }                                                                          \
}


static const char *GetErrorString(sparse_status_t error) {
    switch (error) {
      case SPARSE_STATUS_NOT_INITIALIZED:
        return "SPARSE_STATUS_NOT_INITIALIZED";
      case SPARSE_STATUS_ALLOC_FAILED:
        return "SPARSE_STATUS_ALLOC_FAILED";
      case SPARSE_STATUS_INVALID_VALUE:
        return "SPARSE_STATUS_INVALID_VALUE";
      case SPARSE_STATUS_EXECUTION_FAILED:
        return "SPARSE_STATUS_EXECUTION_FAILED";
      case SPARSE_STATUS_INTERNAL_ERROR:
        return "SPARSE_STATUS_INTERNAL_ERROR";
      case SPARSE_STATUS_NOT_SUPPORTED:
        return "SPARSE_STATUS_NOT_SUPPORTED";
    }
    return "<unknown>";
}


void orthonormalize(fMatrix &A) {
  float *tau = new float[std::max(size_t(1), std::min(A.rows(), A.cols()))];
  CHECK( LAPACKE_sgeqrf(LAPACK_COL_MAJOR, A.rows(), A.cols(), A.data(), A.rows(), tau) );
  CHECK( LAPACKE_sorgqr(LAPACK_COL_MAJOR, A.rows(), A.cols(), A.cols(), A.data(), A.rows(), tau) );
  delete[] tau;
}


void GEMM_SDD(Points &P, fMatrix &R, fMatrix &X) {

  sparse_matrix_t A;
  CHECKS( mkl_sparse_s_create_csr(&A, SPARSE_INDEX_BASE_ZERO, P.rows(), P.cols(), 
        P.rowPtr, P.rowPtr+1, P.colIdx, P.val) );

  matrix_descr descr;
  descr.type = SPARSE_MATRIX_TYPE_GENERAL;

  CHECKS( mkl_sparse_s_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, A, descr, 
        SPARSE_LAYOUT_COLUMN_MAJOR, R.data(), R.cols(), R.rows(), 0., X.data(), X.rows()) );

  CHECKS( mkl_sparse_destroy(A) );
}


void permute_sparse(Points &P, const ivec &perm, dvec &t) {
  
  Timer timer; 

  unsigned n = P.rows(); assert(n == perm.size());
  sparse_matrix_t A, B, C;

  // sparse A for the permutation
  timer.start();
  std::vector<float> A_val(n, 1.0);
  std::vector<int>   A_rowPtr(n+1);
  std::vector<int>   A_colIdx(n);
  timer.stop(); t[8] += timer.elapsed_time();
  
  timer.start();
  par::iota(A_rowPtr.begin(), A_rowPtr.end(), 0);
  cblas_scopy(n, (float*)perm.data(), 1, (float*)A_colIdx.data(), 1);
  timer.stop(); t[7] += timer.elapsed_time();
  
  timer.start();
  CHECKS( mkl_sparse_s_create_csr(&A, SPARSE_INDEX_BASE_ZERO, n, n, 
        A_rowPtr.data(), A_rowPtr.data()+1, A_colIdx.data(), A_val.data()) );

  // sparse B for points 
  CHECKS( mkl_sparse_s_create_csr(&B, SPARSE_INDEX_BASE_ZERO, P.rows(), P.cols(), 
        P.rowPtr, P.rowPtr+1, P.colIdx, P.val) );
  timer.stop(); t[8] += timer.elapsed_time();

  // spmat times spmat
  timer.start();
  CHECKS( mkl_sparse_spmm(SPARSE_OPERATION_NON_TRANSPOSE, A, B, &C) );
  timer.stop(); t[6] += timer.elapsed_time();

  // overwrite input
  timer.start();
  sparse_index_base_t indexing;
  int C_rows, C_cols;
  int *C_rowStart, *C_rowEnd, *C_colIdx;
  float *C_val;
  CHECKS( mkl_sparse_s_export_csr(C, &indexing, &C_rows, &C_cols, &C_rowStart,
        &C_rowEnd, &C_colIdx, &C_val) );
  timer.stop(); t[8] += timer.elapsed_time();

  assert(unsigned(C_rows) == P.rows());
  assert(unsigned(C_cols) == P.cols());
  assert(C_rowEnd[C_rows-1] == P.rowPtr[n]);

  timer.start();
  int nnz = C_rowEnd[C_rows-1];
  cblas_scopy(n, (float *)C_rowStart, 1, (float *)P.rowPtr, 1);
  cblas_scopy(nnz, (float *)C_colIdx, 1, (float *)P.colIdx, 1);
  cblas_scopy(nnz, C_val, 1, P.val, 1);
  timer.stop(); t[7] += timer.elapsed_time();

  timer.start();
  CHECKS( mkl_sparse_destroy(A) );
  CHECKS( mkl_sparse_destroy(B) );
  CHECKS( mkl_sparse_destroy(C) );
  timer.stop(); t[8] += timer.elapsed_time();
}


void transpose_permutation(const ivec &perm, ivec& trans) {
#pragma omp parallel for
  for (size_t i=0; i<perm.size(); i++)
    trans[ perm[i] ] = i;
}


void gather(Points &P, const ivec &perm, dvec &t) {
  permute_sparse(P, perm, t);
}


void scatter(Points &P, const ivec &perm, dvec &t) {
  Timer timer; timer.start();
  ivec trans(perm.size());
  transpose_permutation(perm, trans);
  timer.stop(); t[8] += timer.elapsed_time();
  permute_sparse(P, trans, t);
}


void compute_row_norm(const Points &P, fvec &norm, double &t) {

  sparse_matrix_t A2; // element-wise squared
  fvec A2_val(P.nonZeros()); 
  par::transform(P.val, P.val+P.nonZeros(), A2_val.data(), [](float x){return x*x;});
  
  CHECKS( mkl_sparse_s_create_csr(&A2, SPARSE_INDEX_BASE_ZERO, P.rows(), P.cols(), 
        P.rowPtr, P.rowPtr+1, P.colIdx, A2_val.data()) );

  matrix_descr descr;
  descr.type = SPARSE_MATRIX_TYPE_GENERAL;

  std::vector<float> ones(P.cols());
  par::fill(ones.begin(), ones.end(), 1.0);

  Timer timer; timer.start();
  CHECKS( mkl_sparse_s_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, A2, descr, 
        ones.data(), 0., norm.data()) );
  timer.stop(); t += timer.elapsed_time();

  CHECKS( mkl_sparse_destroy(A2) );
}


void compute_distance(const Points &P, const fvec &norm, fMatrix &D, dvec &t) {
  
  // sparse symmetric rank-k update
  sparse_matrix_t A;
  CHECKS( mkl_sparse_s_create_csr(&A, SPARSE_INDEX_BASE_ZERO, P.rows(), P.cols(), 
        P.rowPtr, P.rowPtr+1, P.colIdx, P.val) );

  // only the upper triangular is computed
  Timer timer; timer.start();
  CHECKS( mkl_sparse_s_syrkd(SPARSE_OPERATION_NON_TRANSPOSE, A, -2, 0, D.data(), 
        SPARSE_LAYOUT_COLUMN_MAJOR, D.rows()) );
  timer.stop(); t[0] += timer.elapsed_time();

  // manually symmetrize the product
  timer.start();
  int n = D.rows();
  for (int i=1; i<n; i++)
    for (int j=0; j<i; j++)
      D(i,j) = D(j,i);
  timer.stop(); t[2] += timer.elapsed_time();

  // rank-1 updates
  timer.start();
  fvec ones(D.rows(), 1.0);
  cblas_sger(CblasColMajor, D.rows(), D.cols(), 1.0, ones.data(), 1, norm.data(), 1, 
      D.data(), D.rows());
  
  cblas_sger(CblasColMajor, D.rows(), D.cols(), 1.0, norm.data(), 1, ones.data(), 1, 
      D.data(), D.rows());
  timer.stop(); t[1] += timer.elapsed_time();

  // free resource 
  CHECKS( mkl_sparse_destroy(A) );
}

/*
void compute_distance(const Points &Q, const Points &R, const fvec &norm, fMatrix &D, dvec &t){
   sparse_matrix_t A;
    CHECKS(mkl_sparse_s_create_csr(&A, SPARSE_INDEX_BASE_ZERO, Q.rows(), Q.cols(), Q.rowPtr, Q.rowPtr+1, Q.colIdx, Q.val) );

   sparse_matrix_t B;

   CHECKS( mkl_sparse_s_create_csr(&B, SPARSE_INDEX_BASE_ZERO, R.rows(), R.cols(), R.rowPtr, R.rowPtr+1, R.colIdx, R.val) );

   Timer timer; timer.start();

   sparse_matrix_t* C;

   CHECKS( mkl_sparse_sp2m(SPARSE_OPERATION_NON_TRANSPOSE,SPARSE_MATRIX_TYPE_GENERAL, A, SPARSE_OPERATION_TRANSPOSE, SPARSE_MATRIX_TYPE_GENERAL, B, SPARSE_STAGE_FULL_MULT, C);

}
*/


void inner_product(const Points &Q, const Points &R, float *D) {
  sparse_matrix_t A, B;
  CHECKS( mkl_sparse_s_create_csr(&A, SPARSE_INDEX_BASE_ZERO, Q.rows(), Q.cols(), 
        Q.rowPtr, Q.rowPtr+1, Q.colIdx, Q.val) );
  CHECKS( mkl_sparse_s_create_csc(&B, SPARSE_INDEX_BASE_ZERO, R.rows(), R.cols(), 
        R.rowPtr, R.rowPtr+1, R.colIdx, R.val) );

  CHECKS( mkl_sparse_s_spmmd(SPARSE_OPERATION_NON_TRANSPOSE, A, B, 
        SPARSE_LAYOUT_ROW_MAJOR, D, R.rows()) );

  CHECKS( mkl_sparse_destroy(A) );
  CHECKS( mkl_sparse_destroy(B) );
}

namespace par{
    void copy(unsigned n, float *src, float *dst) {
      cblas_scopy(n, src, 1, dst, 1);
    }
}

