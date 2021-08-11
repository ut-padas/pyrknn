
#include <stdio.h> 
#include <stdlib.h>
#include <cusparse.h>
#include <cuda.h>
#include <helper_cuda.h>
#include <cuda_runtime.h>








/*
int gen_dense(int m, int d, float *outA)
{
   
  for (i=0; i < N; i++)
  { 
    for (j=0; j < N; j++)
    {
       int r = rand();
*/
       
    
int gen_Spdata(int m, int d, int Nnzperrow, float* V, int* I, int* J)
{
  int i, j;
  //double rMax = (double)RAND_MAX;
  //float *values = (float *)malloc(sizeof(float) * Nnzperrow * m);
  //int *rowptr = (int *)malloc(sizeof(int) *(m+1));
  //int *colInd = (int *)malloc(sizeof(int) * Nnzperrow * m);
  int totalNnz = 0;
    
  I[0] = 0;      
  for (i=0; i<m; i++)
  {
    I[i+1] = I[i] + Nnzperrow;
    for (j=0; j < Nnzperrow; j++) 
    {
      int ind = I[i]+j;
      J[ind] = rand()%d;
      V[ind] = rand();
    }
  }
  
  //V = &
  //*values = V;
  //*rowptr = I;
  //*colInd = J;
  totalNnz += m*Nnzperrow;

  return totalNnz;
}


#define CHECK_CUSPARSE(func) \
{ \
 cusparseStatus_t status = (func); \
 if (status != CUSPARSE_STATUS_SUCCESS) { \
 printf("CUSPARSE API failed at line %d with error: %s (%d)\n", \
 __LINE__, cusparseGetErrorString(status), status); \
 return EXIT_FAILURE; \
 } \
}



int main(int argc, char **argv)
{




  float t_1,t_2,t_3,wtime;


  t_1 = 0;
  t_2 = 0;
  t_3 = 0;
  wtime = 0;



  for (int l=0; l<3000; l++){
  printf(" l = %d , tot_t = %.2f \n", l, wtime);

	//float *dA;
  float *C, *dC;
  //int *dANnzPerRow = 2; 
  float *dCsrValA;
  int *dCsrRowPtrA;
  int *dCsrColIndA;
  //float *dCsrValB;
  //int *dCsrRowPtrB;
  //int *dCsrColIndB;
  float *dCsrValC;
  int *dCsrRowPtrC=0;
  int *dCsrColIndC=0;
  //float *tmp_val;
  //int *tmp_col;
  //int *tmp_row;



  //int *dCNnzPerRow; 
  //int *dCtotalNnz; 
  
  //int totalAnnz_feed;
  //int trueTotalAnnz;

  
  float del_t1;
  float del_t2;
  float del_t3;
  
  checkCudaErrors(cudaSetDevice(0));

  cusparseHandle_t handle = 0; 
  cusparseMatDescr_t Adescr = 0; 
  //cusparseMatDescr_t Bdescr = 0; 
  cusparseMatDescr_t Cdescr = 0; 
  
  cudaEvent_t t0; 
  cudaEvent_t t1;
  cudaEvent_t t2;
  cudaEvent_t t3;

  // trueTotalAnnz = generate_random_dense_matrix(m, d, dANnzPerRow, &A);
  int nnz;
  int m = 300;
  int d = 10000;
  int nnzperrow = 600;
  float *V; 
  int *J; 
  int *I;


  V = (float *)malloc(sizeof(float)*nnzperrow*m);
  J = (int *)malloc(sizeof(int)*nnzperrow*m);
  I = (int *)malloc(sizeof(int)*(m+1));


  int i, j;
  //double rMax = (double)RAND_MAX;
  //float *values = (float *)malloc(sizeof(float) * Nnzperrow * m);
  //int *rowptr = (int *)malloc(sizeof(int) *(m+1));
  //int *colInd = (int *)malloc(sizeof(int) * Nnzperrow * m);

  I[0] = 0;
  for (i=0; i<m; i++)
  {
    I[i+1] = I[i] + nnzperrow;
    for (j=0; j < nnzperrow; j++)
    {
      int ind = I[i]+j;
      J[ind] = rand()%d;
      //float r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX); 
      V[ind] = (float)rand()/(float)(RAND_MAX); 
      //V[ind] = r;
      //printf("J[%d] = %d , V[%d] = %f \n", ind, J[ind], ind, V[ind]);
    }
  }

  //V = &
  //*values = V;
  //*rowptr = I;
  //*colInd = J;
  nnz = m*nnzperrow;








  //nnz = gen_Spdata(m, d, nnzperrow, V, I, J);

  /*
  float values[] = {1, 2, 3, 4, 5, 6};
  int colIdx[] = {0, 2, 2, 0, 1, 2};
  int rowPtr[] = {0, 2, 3, 6};
  int nnz = 6;
  */

 //int size = 6;
  
  //tmp_val = (float *)malloc(nnz*sizeof(float));
  int nnzC;
  //tmp_col = (int *)malloc(nnz*nnz*sizeof(int));
  //tmp_row = (int *)malloc((m+1)*sizeof(int));
  
  C = (float *)malloc(sizeof(float) * m * m); 
  
  
  checkCudaErrors(cusparseCreate(&handle));
 
  // init device arrays

  //checkCudaErrors(cudaMalloc((void **)&dA, sizeof(float) * m * d));
  checkCudaErrors(cudaMalloc((void **)&dC, sizeof(float) * m * m));

  // descriptors
  checkCudaErrors(cusparseCreateMatDescr(&Adescr));
  //checkCudaErrors(cusparseCreateMatDescr(&Bdescr));
  checkCudaErrors(cusparseCreateMatDescr(&Cdescr));
  
  checkCudaErrors(cusparseSetMatType(Adescr, CUSPARSE_MATRIX_TYPE_GENERAL));
  //checkCudaErrors(cusparseSetMatType(Bdescr, CUSPARSE_MATRIX_TYPE_GENERAL));
  checkCudaErrors(cusparseSetMatType(Cdescr, CUSPARSE_MATRIX_TYPE_GENERAL));
  
  checkCudaErrors(cusparseSetMatIndexBase(Adescr, CUSPARSE_INDEX_BASE_ZERO));
  //checkCudaErrors(cusparseSetMatIndexBase(Bdescr, CUSPARSE_INDEX_BASE_ZERO));
  checkCudaErrors(cusparseSetMatIndexBase(Cdescr, CUSPARSE_INDEX_BASE_ZERO));

  // init csr format for input A and output C
  checkCudaErrors(cudaMalloc((void **)&dCsrValA, sizeof(float) * nnz));
  checkCudaErrors(cudaMalloc((void **)&dCsrRowPtrA, sizeof(int) * (m+1)));
  checkCudaErrors(cudaMalloc((void **)&dCsrColIndA, sizeof(int) * nnz));

  //checkCudaErrors(cudaMalloc((void **)&dCsrValB, sizeof(float) * nnz));
  //checkCudaErrors(cudaMalloc((void **)&dCsrRowPtrB, sizeof(int) * (m+1)));
  //checkCudaErrors(cudaMalloc((void **)&dCsrColIndB, sizeof(int) * nnz));

  checkCudaErrors(cudaMalloc((void **)&dCsrRowPtrC, sizeof(int) * (m+1)));
  //checkCudaErrors(cudaMalloc((void **)&dC, sizeof(float) * m * m));

  // copy input to device
  //checkCudaErrors(cudaMemcpy(&dA, A, sizeof(float) * m * d, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(dCsrValA, V, sizeof(float) * nnz, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(dCsrRowPtrA, I, sizeof(int) * (m+1), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(dCsrColIndA, J, sizeof(int) * nnz, cudaMemcpyHostToDevice));
  
  //checkCudaErrors(cudaMemcpy(dCsrValB, values, sizeof(float) * nnz, cudaMemcpyHostToDevice));
  //checkCudaErrors(cudaMemcpy(dCsrRowPtrB, rowPtr, sizeof(int) * (m+1), cudaMemcpyHostToDevice));
  //checkCudaErrors(cudaMemcpy(dCsrColIndB, colIdx, sizeof(int) * nnz, cudaMemcpyHostToDevice));

  // timer  
  checkCudaErrors(cudaEventCreate(&t0));
  checkCudaErrors(cudaEventCreate(&t1));
  checkCudaErrors(cudaEventCreate(&t2)); 
  checkCudaErrors(cudaEventCreate(&t3)); 
  


  // set the device output array init value 
  //checkCudaErrors(cudaMemset(dC, 0, sizeof(float) * m * m));

  // get the number of nonzeros for dense2csr
  //checkCudaErrors(cusparseSnnz(handle, CUSPARSE_DIRECTION_ROW, m, m, Adescr, dA, m, dANnzPerRow, &trueTotalAnnz)); 

  int *nnzTotalDevHostPtr = &nnzC;
  checkCudaErrors(cudaEventRecord(t0, 0));
  
  checkCudaErrors(cusparseXcsrgemmNnz(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE, m, m, d,
                                      Adescr, nnz, dCsrRowPtrA, dCsrColIndA, 
                                      Adescr, nnz, dCsrRowPtrA, dCsrColIndA,
                                      Cdescr, dCsrRowPtrC, nnzTotalDevHostPtr));

  nnzC = *nnzTotalDevHostPtr;
  
  
  checkCudaErrors(cudaMalloc((void **)&dCsrValC, sizeof(float) * nnzC));
  checkCudaErrors(cudaMalloc((void **)&dCsrColIndC, sizeof(int) * nnzC));
                                       
                                   
  // transfer the dense 2 csr
  //checkCudaErrors(cusparseSdense2csr(handle, m, d, Adescr, dA, m, dANnzPerRow, dCsrValA, dCsrRowPtrA, dCsrColIndA));
 
  // timer
  checkCudaErrors(cudaEventRecord(t1, 0));
  checkCudaErrors(cudaEventSynchronize(t1)); 
  checkCudaErrors(cudaEventElapsedTime(&del_t1, t0, t1));
  
  //checkCudaErrors(cudaMalloc((void **)&dCsrRowPtrC, sizeof(int) * (m+1)));
  checkCudaErrors(cudaMemcpy(dC, C, sizeof(float) * m *m, cudaMemcpyHostToDevice));
  // SpGeMM 
  checkCudaErrors(cusparseScsrgemm(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE, m, m, d, 
                                  Adescr, nnz, dCsrValA, dCsrRowPtrA, dCsrColIndA, 
                                  Adescr, nnz, dCsrValA, dCsrRowPtrA, dCsrColIndA, 
                                  Cdescr, dCsrValC, dCsrRowPtrC, dCsrColIndC)); 
 
  // timer
  checkCudaErrors(cudaEventRecord(t2, 0));
  checkCudaErrors(cudaEventSynchronize(t2)); 
  cudaDeviceSynchronize();
  checkCudaErrors(cudaEventElapsedTime(&del_t2, t1, t2));


  //checkCudaErrors(cudaMemcpy(tmp_col, dCsrRowPtrC, sizeof(int) * (m+1), cudaMemcpyDeviceToHost)); 
  //tmp_val = (float *)malloc(sizeof(float)*nnzC);
  //checkCudaErrors(cudaMemcpy(tmp_val, dCsrValC, sizeof(float)*nnzC, cudaMemcpyDeviceToHost)); 
  
     

  //checkCudaErrors(cudaMemcpy(tmp_row, dCsrRowPtrC, sizeof(int) * (m+1), cudaMemcpyDeviceToHost)); 



  // csr 2 dense 
  checkCudaErrors(cusparseScsr2dense(handle, m, m, Cdescr, dCsrValC, dCsrRowPtrC, dCsrColIndC, dC, m));
  
  checkCudaErrors(cudaEventRecord(t3, 0));
  checkCudaErrors(cudaEventSynchronize(t3)); 
  checkCudaErrors(cudaEventElapsedTime(&del_t3, t2, t3));
  

  //measure nonzeros
	//checkCudaErrors(cusparseSnnz(handle, CUSPARSE_DIRECTION_ROW, m,m, Cdescr, dC, m, dCNnzPerRow, dCtotalNnz)); 
  
  // device to host                    
  checkCudaErrors(cudaMemcpy(C, dC, sizeof(int)* m * m, cudaMemcpyDeviceToHost));


  // measure time 

  // output
  float tot_t = del_t1 + del_t2 + del_t3;
  
  t_1 += del_t1;
  t_2 += del_t2;
  t_3 += del_t3;
  wtime += tot_t;


  /*
  for (int i = 0; i < nnz; i++)
  { 
      printf(" %4f ", V[i]);
  }
  printf(" \n\n");

  for (int i = 0; i < m; i++)
  { 
    for (int j= 0; j < m; j++)
    {
      printf(" %f ", C[i*m + j]);
    }
    printf(" \n");
  }
  
  */

  free(C);
  free(V);
  free(I);
  free(J);
  //free(A);
  //checkCudaErrors(cudaFree(dA));
  //free(values);
  //free(rowPtr);
  //free(colIdx);
  checkCudaErrors(cudaFree(dCsrValA));
  checkCudaErrors(cudaFree(dCsrRowPtrA));
  checkCudaErrors(cudaFree(dCsrColIndA));
  /* 
  checkCudaErrors(cudaFree(dCsrValB));
  checkCudaErrors(cudaFree(dCsrRowPtrB));
  checkCudaErrors(cudaFree(dCsrColIndB));
  */
  checkCudaErrors(cudaFree(dCsrValC));
  checkCudaErrors(cudaFree(dCsrRowPtrC));
  checkCudaErrors(cudaFree(dCsrColIndC));
  checkCudaErrors(cudaFree(dC));
  //checkCudaErrors(cudaEventDestroy(t0));
  //checkCudaErrors(cudaEventDestroy(t1));
  //checkCudaErrors(cudaEventDestroy(t2));
  checkCudaErrors(cusparseDestroyMatDescr(Adescr));
  checkCudaErrors(cusparseDestroyMatDescr(Cdescr));
	checkCudaErrors(cusparseDestroy(handle));

  }
  printf("Get nnz of output : %.2f \n", t_1); 
  printf("SpGeMM : %.2f \n", t_2); 
  printf("csr to dense : %.2f \n", t_3); 
  printf("total time : %.2f \n", wtime);
  return 0;
}

