#include "util_gpu.hpp"

struct column_idx: public thrust::unary_function<int, int> {
  int nCol;

  __host__ __device__
  column_idx(int c): nCol(c) {}

  __host__ __device__
  int operator()(int i) {
    return i%nCol;
  }
};


void gemv_gpu(const int *hA_csrOffsets, const int *hA_columns, const float *hA_values, 
    int A_num_rows, int A_num_cols, int A_num_nnz, const float *hX, float *hY) {
    // y = A * x
    float alpha = 1.0f;
    float beta  = 0.0f;
    // Device memory management
    int   *dA_csrOffsets, *dA_columns;
    float *dA_values, *dX, *dY;
    CHECK_CUDA( cudaMalloc((void**) &dA_csrOffsets,
                           (A_num_rows + 1) * sizeof(int)) )
    CHECK_CUDA( cudaMalloc((void**) &dA_columns, A_num_nnz * sizeof(int)) )
    CHECK_CUDA( cudaMalloc((void**) &dA_values, A_num_nnz * sizeof(float)) )
    CHECK_CUDA( cudaMalloc((void**) &dX, A_num_cols * sizeof(float)) )
    CHECK_CUDA( cudaMalloc((void**) &dY, A_num_rows * sizeof(float)) )

    CHECK_CUDA( cudaMemcpy(dA_csrOffsets, hA_csrOffsets,
                           (A_num_rows + 1) * sizeof(int),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dA_columns, hA_columns, A_num_nnz * sizeof(int),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dA_values, hA_values,
                           A_num_nnz * sizeof(float), cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dX, hX, A_num_cols * sizeof(float),
                           cudaMemcpyHostToDevice) )  
    //--------------------------------------------------------------------------
    // CUSPARSE APIs
    cusparseHandle_t     handle = 0;
    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vecX, vecY;
    void*  dBuffer    = NULL;
    size_t bufferSize = 0;
    CHECK_CUSPARSE( cusparseCreate(&handle) )
    // Create sparse matrix A in CSR format
    CHECK_CUSPARSE( cusparseCreateCsr(&matA, A_num_rows, A_num_cols, A_num_nnz,
                                      dA_csrOffsets, dA_columns, dA_values,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) )
    // Create dense vector X
    CHECK_CUSPARSE( cusparseCreateDnVec(&vecX, A_num_cols, dX, CUDA_R_32F) )
    // Create dense vector y
    CHECK_CUSPARSE( cusparseCreateDnVec(&vecY, A_num_rows, dY, CUDA_R_32F) )
    // allocate an external buffer if needed
    CHECK_CUSPARSE( cusparseSpMV_bufferSize(
                                 handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matA, vecX, &beta, vecY, CUDA_R_32F,
                                 CUSPARSE_MV_ALG_DEFAULT, &bufferSize) )
    CHECK_CUDA( cudaMalloc(&dBuffer, bufferSize) )

    // execute SpMV
    CHECK_CUSPARSE( cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matA, vecX, &beta, vecY, CUDA_R_32F,
                                 CUSPARSE_MV_ALG_DEFAULT, dBuffer) )

    // destroy matrix/vector descriptors
    CHECK_CUSPARSE( cusparseDestroySpMat(matA) )
    CHECK_CUSPARSE( cusparseDestroyDnVec(vecX) )
    CHECK_CUSPARSE( cusparseDestroyDnVec(vecY) )
    CHECK_CUSPARSE( cusparseDestroy(handle) )
    //--------------------------------------------------------------------------
    // device result check
    CHECK_CUDA( cudaMemcpy(hY, dY, A_num_rows * sizeof(float),
                           cudaMemcpyDeviceToHost) )
    //--------------------------------------------------------------------------
    // device memory deallocation
    CHECK_CUDA( cudaFree(dBuffer) )
    CHECK_CUDA( cudaFree(dA_csrOffsets) )
    CHECK_CUDA( cudaFree(dA_columns) )
    CHECK_CUDA( cudaFree(dA_values) )
    CHECK_CUDA( cudaFree(dX) )
    CHECK_CUDA( cudaFree(dY) )
}


int sum(const int *vec, int size) {
  int s = 0;
  for (int i=0; i<size; i++)
    s += vec[i];
  return s;
}


void copy_and_shift(int *dst, const int *src, int size, int shift) {
  for (int i=0; i<size; i++)
    dst[i] = src[i] + shift;
}


void create_BDSpMat(int *hA_rowPtr[], int *hA_colIdx[], float *hA_val[], 
    int *M, int d, int *NNZ, int nLeaf,
    int *rowPtrA, int *colIdxA, float *valA) {
  
  int m = sum(M, nLeaf), n = d*nLeaf, nnz = sum(NNZ, nLeaf);
  int cum_nnz = 0, cum_row = 0, cum_col = 0;
  for (int i=0; i<nLeaf; i++) {
    std::memcpy(valA+cum_nnz, hA_val[i], NNZ[i]*sizeof(float));
    copy_and_shift(rowPtrA+cum_row, hA_rowPtr[i], M[i]+1, cum_nnz);
    copy_and_shift(colIdxA+cum_nnz, hA_colIdx[i], NNZ[i], cum_col);
    
    cum_nnz += NNZ[i];
    cum_row += M[i];
    cum_col += d;
  }
  assert(cum_nnz == nnz);
  assert(cum_row == m);
  assert(cum_col == n);
}


// (sparse) A * (sparse) B = (sparse) C
void GEMM_SSS(int m, int n, int k, float alpha,
    int *csrRowPtrA, int *csrColIndA, float *csrValA, int nnzA,
    int *csrRowPtrB, int *csrColIndB, float *csrValB, int nnzB,
    int* &csrRowPtrC, int* &csrColIndC, float* &csrValC, int &nnzC) {
  

  // step 1: create an opaque structure
  csrgemm2Info_t info = NULL;
  CHECK_CUSPARSE( cusparseCreateCsrgemm2Info(&info) )

  // step 2: allocate buffer for csrgemm2Nnz and csrgemm2
  cusparseHandle_t handle;
  CHECK_CUSPARSE( cusparseCreate(&handle) )
  
  cusparseMatDescr_t descr;
  CHECK_CUSPARSE( cusparseCreateMatDescr(&descr) )
  CHECK_CUSPARSE( cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL) )
  CHECK_CUSPARSE( cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO) )
  // dummy matrix D
  int nnzD = 0, csrRowPtrD[m+1] = {0}, *csrColIndD = 0;
  float *csrValD = 0;
  
  size_t bufferSize;
  CHECK_CUSPARSE( cusparseScsrgemm2_bufferSizeExt(
        handle, m, n, k, &alpha,
        descr, nnzA, csrRowPtrA, csrColIndA,
        descr, nnzB, csrRowPtrB, csrColIndB,
        NULL,
        descr, nnzD, csrRowPtrD, csrColIndD,
        info,
        &bufferSize) )

  void *buffer = NULL;
  CHECK_CUDA( cudaMalloc(&buffer, bufferSize) )

  // step 3: compute csrRowPtrC
  CHECK_CUDA( cudaMalloc((void**)&csrRowPtrC, sizeof(int)*(m+1)) )
  int *nnzTotalDevHostPtr = &nnzC;
  cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST);
  CHECK_CUSPARSE( cusparseXcsrgemm2Nnz(handle, m, n, k,
          descr, nnzA, csrRowPtrA, csrColIndA,
          descr, nnzB, csrRowPtrB, csrColIndB,
          descr, nnzD, csrRowPtrD, csrColIndD,
          descr, csrRowPtrC, nnzTotalDevHostPtr,
          info, buffer) )
  if (NULL != nnzTotalDevHostPtr){
      nnzC = *nnzTotalDevHostPtr;
  }else{
      int baseC;
      cudaMemcpy(&nnzC, csrRowPtrC+m, sizeof(int), cudaMemcpyDeviceToHost);
      cudaMemcpy(&baseC, csrRowPtrC, sizeof(int), cudaMemcpyDeviceToHost);
      nnzC -= baseC;
  }  

  // step 4: finish sparsity pattern and value of C
  // Remark: set csrValC to null if only sparsity pattern is required 
  CHECK_CUDA( cudaMalloc((void**)&csrColIndC, sizeof(int)*nnzC) )
  CHECK_CUDA( cudaMalloc((void**)&csrValC, sizeof(double)*nnzC) )
  CHECK_CUSPARSE( cusparseScsrgemm2(handle, m, n, k, &alpha,
          descr, nnzA, csrValA, csrRowPtrA, csrColIndA,
          descr, nnzB, csrValB, csrRowPtrB, csrColIndB,
          NULL,
          descr, nnzD, csrValD, csrRowPtrD, csrColIndD,
          descr,       csrValC, csrRowPtrC, csrColIndC,
          info, buffer) )

  // step 5: clean up
  CHECK_CUSPARSE( cusparseDestroyCsrgemm2Info(info) )
  CHECK_CUSPARSE( cusparseDestroy(handle) )
  CHECK_CUSPARSE( cusparseDestroyMatDescr(descr) )
  
  if (buffer != NULL) CHECK_CUDA( cudaFree(buffer) )
}


// (sparse) A * (sparse) B = (dense) C
void GEMM_SSD(int m, int n, int k, float alpha,
    int *csrRowPtrA, int *csrColIndA, float *csrValA, int nnzA,
    int *csrRowPtrB, int *csrColIndB, float *csrValB, int nnzB,
    int *csrRowPtrC, int *csrColIndC, float *csrValC, int nnzC) {
  

  // step 1: create an opaque structure
  csrgemm2Info_t info = NULL;
  CHECK_CUSPARSE( cusparseCreateCsrgemm2Info(&info) )

  // step 2: allocate buffer for csrgemm2Nnz and csrgemm2
  cusparseHandle_t handle;
  CHECK_CUSPARSE( cusparseCreate(&handle) )
  
  cusparseMatDescr_t descr;
  CHECK_CUSPARSE( cusparseCreateMatDescr(&descr) )
  CHECK_CUSPARSE( cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL) )
  CHECK_CUSPARSE( cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO) )
  // dummy matrix D
  int nnzD = 0, csrRowPtrD[m+1] = {0}, *csrColIndD = 0;
  float *csrValD = 0;
  
  size_t bufferSize;
  CHECK_CUSPARSE( cusparseScsrgemm2_bufferSizeExt(
        handle, m, n, k, &alpha,
        descr, nnzA, csrRowPtrA, csrColIndA,
        descr, nnzB, csrRowPtrB, csrColIndB,
        NULL,
        descr, nnzD, csrRowPtrD, csrColIndD,
        info,
        &bufferSize) )

  void *buffer = NULL;
  CHECK_CUDA( cudaMalloc(&buffer, bufferSize) )

  // step 3: compute csrRowPtrC
  // csrRowPtrC and nnzC are inputs

  // step 4: finish sparsity pattern and value of C
  // Remark: set csrValC to null if only sparsity pattern is required 
  CHECK_CUSPARSE( cusparseScsrgemm2(handle, m, n, k, &alpha,
          descr, nnzA, csrValA, csrRowPtrA, csrColIndA,
          descr, nnzB, csrValB, csrRowPtrB, csrColIndB,
          NULL,
          descr, nnzD, csrValD, csrRowPtrD, csrColIndD,
          descr,       csrValC, csrRowPtrC, csrColIndC,
          info, buffer) )

  // step 5: clean up
  CHECK_CUSPARSE( cusparseDestroyCsrgemm2Info(info) )
  CHECK_CUSPARSE( cusparseDestroy(handle) )
  CHECK_CUSPARSE( cusparseDestroyMatDescr(descr) )
  
  if (buffer != NULL) CHECK_CUDA( cudaFree(buffer) )
}


/*
 * Assume: indices are 0-based, so the row pointer always starts with a 0.
 */
void batchedGemv_gpu(int *hA_rowPtr[], int *hA_colIdx[], float *hA_val[], 
    int *M, int d, int *NNZ, float *hX[], float *hY[], int nLeaf) {

    // create three block-diagonal matrices on GPU
    int   *dA_rowPtr, *dA_colIdx, *dX_rowPtr, *dX_colIdx, *dY_rowPtr, *dY_colIdx;
    float *dA_val, *dX_val, *dY_val;
    
    int m = sum(M, nLeaf), n = d*nLeaf, nnz = sum(NNZ, nLeaf);

    CHECK_CUDA( cudaMalloc((void**) &dA_rowPtr, (m + 1) * sizeof(int)) )
    CHECK_CUDA( cudaMalloc((void**) &dA_colIdx, nnz * sizeof(int)) )
    CHECK_CUDA( cudaMalloc((void**) &dA_val,    nnz * sizeof(float)) )

    CHECK_CUDA( cudaMalloc((void**) &dX_rowPtr, (n + 1) * sizeof(int)) )
    CHECK_CUDA( cudaMalloc((void**) &dX_colIdx, n * sizeof(int)) )
    CHECK_CUDA( cudaMalloc((void**) &dX_val,    n * sizeof(float)) )

    CHECK_CUDA( cudaMalloc((void**) &dY_rowPtr, (m + 1) * sizeof(int)) )
    CHECK_CUDA( cudaMalloc((void**) &dY_colIdx, m * sizeof(int)) )
    CHECK_CUDA( cudaMalloc((void**) &dY_val,    m * sizeof(float)) )

    // pointers for A
    int rowPtrA[m+1], colIdxA[nnz];
    float valA[nnz];
    create_BDSpMat(hA_rowPtr, hA_colIdx, hA_val, M, d, NNZ, nLeaf,
        rowPtrA, colIdxA, valA);
    //print(m, n, nnz, rowPtrA, colIdxA, valA);
    

    CHECK_CUDA( cudaMemcpy(dA_rowPtr, rowPtrA, (m+1) * sizeof(int), cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dA_colIdx, colIdxA, nnz * sizeof(int), cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dA_val, valA, nnz * sizeof(float), cudaMemcpyHostToDevice) )

    // pointers for X
    int rowPtrX[n+1], colIdxX[n];
    float valX[n];
    std::iota(rowPtrX, rowPtrX+n+1, 0); // one nonzero per row
    for (int i=0; i<nLeaf; i++) {
      std::memcpy(valX+i*d, hX[i], d*sizeof(float));
      std::fill_n(colIdxX+i*d, d, i);
    }
    //print(n, nLeaf, n, rowPtrX, colIdxX, valX);


    CHECK_CUDA( cudaMemcpy(dX_rowPtr, rowPtrX, (n+1) * sizeof(int), cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dX_colIdx, colIdxX, n * sizeof(int), cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dX_val, valX, n * sizeof(float), cudaMemcpyHostToDevice) )

    // pointers for Y
    int rowPtrY[m+1];
    std::iota(rowPtrY, rowPtrY+m+1, 0); // one nonzero per row
    CHECK_CUDA( cudaMemcpy(dY_rowPtr, rowPtrY, (m+1) * sizeof(int), cudaMemcpyHostToDevice) )


    // compute Y
    GEMM_SSD(m, nLeaf, n, 1.0, 
        dA_rowPtr, dA_colIdx, dA_val, nnz,
        dX_rowPtr, dX_colIdx, dX_val, n,
        dY_rowPtr, dY_colIdx, dY_val, m);
    

    // copy results to host
    float hY_val[m];
    CHECK_CUDA( cudaMemcpy(hY_val, dY_val, m * sizeof(float), cudaMemcpyDeviceToHost) )
    float *hY_cum = hY_val;
    for (int i=0; i<nLeaf; i++) {
      std::copy(hY_cum, hY_cum+M[i], hY[i]);
      hY_cum += M[i];
    }

    // free memory
    CHECK_CUDA( cudaFree(dA_rowPtr) ); 
    CHECK_CUDA( cudaFree(dA_colIdx) );
    CHECK_CUDA( cudaFree(dA_val) );
    
    CHECK_CUDA( cudaFree(dX_rowPtr) ); 
    CHECK_CUDA( cudaFree(dX_colIdx) );
    CHECK_CUDA( cudaFree(dX_val) );
    
    CHECK_CUDA( cudaFree(dY_rowPtr) ); 
    CHECK_CUDA( cudaFree(dY_colIdx) );
    CHECK_CUDA( cudaFree(dY_val) );
}


void compute_distance_gpu(int *hP_rowPtr[], int *hP_colIdx[], float *hP_val[], 
    int *M, int d, int *NNZ, float *hD, int nLeaf, int b, bool debug) {

  int m = sum(M, nLeaf), n = d*nLeaf, nnz = sum(NNZ, nLeaf);
  int rowPtrP[m+1], colIdxP[nnz];
  float valP[nnz];
  create_BDSpMat(hP_rowPtr, hP_colIdx, hP_val, M, d, NNZ, nLeaf,
      rowPtrP, colIdxP, valP);

  if (debug) print(m, n, nnz, rowPtrP, colIdxP, valP, "Points");

  // allocate/copy data on/to device
  int *dP_rowPtr, *dP_colIdx, *dR_colPtr, *dR_rowIdx;
  float *dP_val, *dR_val;

  CHECK_CUDA( cudaMalloc((void**) &dP_rowPtr, (m + 1) * sizeof(int)) )
  CHECK_CUDA( cudaMalloc((void**) &dP_colIdx, nnz * sizeof(int)) )
  CHECK_CUDA( cudaMalloc((void**) &dP_val,    nnz * sizeof(float)) )

  CHECK_CUDA( cudaMemcpy(dP_rowPtr, rowPtrP, (m+1) * sizeof(int), cudaMemcpyHostToDevice) )
  CHECK_CUDA( cudaMemcpy(dP_colIdx, colIdxP, nnz * sizeof(int), cudaMemcpyHostToDevice) )
  CHECK_CUDA( cudaMemcpy(dP_val,    valP,    nnz * sizeof(float), cudaMemcpyHostToDevice) )

  CHECK_CUDA( cudaMalloc((void**) &dR_colPtr, (n + 1) * sizeof(int)) )
  CHECK_CUDA( cudaMalloc((void**) &dR_rowIdx, nnz * sizeof(int)) )
  CHECK_CUDA( cudaMalloc((void**) &dR_val,    nnz * sizeof(float)) )

  // R = Q^T
  cusparseHandle_t handle;
  CHECK_CUSPARSE( cusparseCreate(&handle) )

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

  if (debug) dprint(n, m, nnz, dR_colPtr, dR_rowIdx, dR_val, "Transpose");


  // compute row norms
  dvec<float> P2(m, 0.);
  dvec<int> Prow(m), rowIdx(nnz);
  for (int i=0; i<m; i++) {
    thrust::fill_n(rowIdx.begin()+rowPtrP[i], rowPtrP[i+1]-rowPtrP[i], i);
  }

  dptr<float> ptrP(dP_val);
  auto iterP2 = thrust::make_transform_iterator(ptrP, thrust::square<float>());
  auto end = thrust::reduce_by_key(rowIdx.begin(), rowIdx.end(), iterP2, Prow.begin(), P2.begin());

  if (debug) dprint(m, thrust::raw_pointer_cast(Prow.data()), "P2 row indices");
  
  // handle zero rows
  dvec<float> P2_tmp;
  int s = end.first - Prow.begin();
  if (s < m) {
    P2_tmp.resize(m, 0.);
    auto perm = thrust::make_permutation_iterator(P2_tmp.begin(), Prow.begin());
    thrust::copy(P2.begin(), P2.begin()+s, perm);
    P2 = P2_tmp;
  }

  if (debug) dprint(m, thrust::raw_pointer_cast(P2.data()), "GPU point norm");
  

  // loop over blocks
  assert(M[0]%b == 0); // TODO: assume same number of rows
  int nBlock = M[0]/b;
  for (int i=0; i<nBlock; i++) {
    // extract strided block
    int rowPtrQ[b*nLeaf+1]; // b*nLeaf rows
    int cum_nnz = 0;
    for (int j=0; j<nLeaf; j++) {
      copy_and_shift(rowPtrQ+j*b, rowPtrP+j*M[0]+i*b, b+1, cum_nnz-rowPtrP[j*M[0]+i*b]);
      cum_nnz = rowPtrQ[(j+1)*b];
    }
    int colIdxQ[cum_nnz];
    float valQ[cum_nnz];
    for (int j=0; j<nLeaf; j++) {
      int sq = rowPtrQ[j*b];
      int sp = rowPtrP[j*M[0]+i*b];
      int block_nnz = rowPtrQ[(j+1)*b] - sq;
      assert(block_nnz == rowPtrP[j*M[0]+(i+1)*b] - sp);
      std::memcpy(colIdxQ+sq, colIdxP+sp, block_nnz*sizeof(int));
      std::memcpy(valQ+sq, valP+sp, block_nnz*sizeof(float));
    }
    if (debug) print(b*nLeaf, n, cum_nnz, rowPtrQ, colIdxQ, valQ, "Query");
   
    // copy Q to device
    int *dQ_rowPtr, *dQ_colIdx;
    float *dQ_val;
    
    CHECK_CUDA( cudaMalloc((void**) &dQ_rowPtr, (b*nLeaf + 1) * sizeof(int)) )
    CHECK_CUDA( cudaMalloc((void**) &dQ_colIdx, cum_nnz * sizeof(int)) )
    CHECK_CUDA( cudaMalloc((void**) &dQ_val,    cum_nnz * sizeof(float)) )

    CHECK_CUDA( cudaMemcpy(dQ_rowPtr, rowPtrQ, (b*nLeaf+1) * sizeof(int), cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dQ_colIdx, colIdxQ, cum_nnz * sizeof(int), cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dQ_val,    valQ,    cum_nnz * sizeof(float), cudaMemcpyHostToDevice) )
      
    // setup for result
    int *dD_rowPtr, *dD_colIdx, nnzD;
    float *dD_val;

    GEMM_SSS(b*nLeaf, M[0]*nLeaf, d*nLeaf, -2, 
        dQ_rowPtr, dQ_colIdx, dQ_val, cum_nnz,
        dR_colPtr, dR_rowIdx, dR_val, nnz,
        dD_rowPtr, dD_colIdx, dD_val, nnzD);
  
    if (debug) dprint(b*nLeaf, M[0]*nLeaf, nnzD, dD_rowPtr, dD_colIdx, dD_val, "GEMM");

    // convert to dense matrix
    float *Dist;
    CHECK_CUDA( cudaMalloc((void**) &Dist, b*nLeaf*M[0] * sizeof(float)) )

    cusparseMatDescr_t descr;
    CHECK_CUSPARSE( cusparseCreateMatDescr(&descr) )
    CHECK_CUSPARSE( cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL) )
    CHECK_CUSPARSE( cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO) )

    dvec<int> dense_colIdx(nnzD);
    dptr<int> sparse_colIdx(dD_colIdx);
    thrust::transform(sparse_colIdx, sparse_colIdx+nnzD, 
        dense_colIdx.begin(), column_idx(M[0]));
    int *den_colIdx = thrust::raw_pointer_cast(dense_colIdx.data());

    // treat as a csc format (notice the order of inputs are different from csr2dense)
    // output is a M[0]-by-b*nLeaf matrix in column-major
    CHECK_CUSPARSE( cusparseScsc2dense(
          handle, M[0], b*nLeaf,
          descr, dD_val, den_colIdx, dD_rowPtr,
          Dist, M[0]) )


    // rank-1 updates
    int oneInt = 1; float oneFloat = 1.;
    dvec<float> ones(m, 1.0); 
    float *ptrOne = thrust::raw_pointer_cast(ones.data());
    float *ptrP2 = thrust::raw_pointer_cast(P2.data());

    cublasHandle_t hCublas;
    CHECK_CUBLAS( cublasCreate(&hCublas) )

    CHECK_CUBLAS( cublasSgemmStridedBatched(
          hCublas, CUBLAS_OP_N, CUBLAS_OP_T, 
          M[0], b, oneInt, &oneFloat, 
          ptrP2, M[0], M[0],
          ptrOne, b, b,
          &oneFloat, Dist, M[0], b*M[0], nLeaf) );

    CHECK_CUBLAS( cublasSgemmStridedBatched(
          hCublas, CUBLAS_OP_N, CUBLAS_OP_T, 
          M[0], b, oneInt, &oneFloat, 
          ptrOne, M[0], M[0],
          ptrP2+i*b, b, M[0],
          &oneFloat, Dist, M[0], b*M[0], nLeaf) );


    // copy results to host
    float hD_block[b*nLeaf*M[0]];
    CHECK_CUDA( cudaMemcpy(hD_block, Dist, (b*nLeaf*M[0])*sizeof(float), cudaMemcpyDeviceToHost) )

    if (debug) print(b*nLeaf, M[0], hD_block, "Dense block");
    //print(M[0], b*nLeaf, hD_block, "Dense block");

    for (int j=0; j<nLeaf; j++) {
      std::memcpy(hD+(i*b+j*M[0])*M[0], hD_block+j*b*M[0], b*M[0]*sizeof(float));
    }
  
    // clean up
    CHECK_CUSPARSE( cusparseDestroyMatDescr(descr) )
  }
}


