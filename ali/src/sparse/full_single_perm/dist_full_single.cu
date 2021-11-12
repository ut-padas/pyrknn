#define SM_SIZE_11 2048
#define SM_SIZE_12 4096
#define SM_SIZE_13 8192


#include "dist_full_single.h"


__device__ int BinSearch(int* arr, int elem, int start, int stop, int shift, int init, int delay_par){

  int ret = start;
  int testInd;

  for (int l = stop - start; l > 1; l -= floorf(l/2.0)){

    testInd = (ret + l < stop - 1) ? ret + l : stop -1;
    ret = (arr[shift + testInd] <= elem) ? testInd : ret;

  }
  
  testInd = (ret + 1 < stop - 1) ? ret + 1 : stop - 1;
  ret = (arr[shift + testInd] <= elem) ? testInd : ret;
  
  return ret;
}


__global__ void ComputeDists_seq(int* R_ref, int* C_ref, float* V_ref, int* R_q, int* C_q, float* V_q, int* local_leafIds, float* Norms_q, float* Norms_ref, int const k_nn, float* KNN_tmp, int const ppl, int* QId, int* glob_pointIds){
  

  __shared__ int SM_C_q [SM_SIZE_13];

  int tid = threadIdx.x;

  int q = blockIdx.x;

  int ind0_q = R_q[q];
  int ind1_q = R_q[q+1];
  int nnz_q = ind1_q - ind0_q;
  float norm_q = Norms_q[q];
  
  // avoid bank conflict
  /*
  int num_copies = SM_SIZE_13/nnz_q;
  int batchsize = blockDim.x / num_copies;
  if (batchsize * num_copies < blockDim.x) batchsize += 1; 
  int batch = tid / batchsize;
  int loop_shift = (blockDim.x < (batch + 1) * batchsize) ? blockDim.x - batch * batchsize : batchsize;
  */
  //int shift = batch * nnz_q;

  int shift = 0; 

  for (int n = tid; n < nnz_q; n += blockDim.x){
    SM_C_q[n] = C_q[ind0_q + n];
    //for (int n = tid - batch * batchsize; n < nnz_q; n += loop_shift) {
    //SM_C_q[shift + n] = C_q[ind0_q + n];
    //if (QId[q] == 5) printf("C[%d] = %d \n", n, C_q[ind0_q + n]); 
  }
  __syncthreads(); 
  

  int leafId = local_leafIds[q];
  int nq = gridDim.x;
  float c_tmp = 0.0; 
  //int init = tid % 5;
  int init = 0;
  int testInd;
  int delay_par = 0;
  for (int pt = tid; pt < ppl; pt += blockDim.x){
    
    c_tmp = 0.0;
    int ptId = leafId * ppl + pt;
    if (QId[q] == 678 && glob_pointIds[ptId] == 1303829){
    //if (QId[q] == 5){
      //printf("glob_id [%d] = %d \n", ptId, glob_pointIds[ptId]);
      /* 
      for (int n = 0; n < nnz_q; n++){
        printf("C[%d] = %d \n", n, SM_C_q[n]);
      }
      */
      
    }

    

    int ind0_pt = R_ref[ptId];
    int nnz_pt = R_ref[ptId+1] - ind0_pt;
    int ret = 0;
  
    for (int pos_k = 0; pos_k < nnz_pt; pos_k++){
      
      int k = C_ref[ind0_pt + pos_k];
      init = (nnz_q - ret < init) ? 0 : init;
      
      // make a delay 
      if (tid > 200 && tid > 400 && tid > 600 && tid > 800) delay_par = 1; 
  
      ret = BinSearch(SM_C_q, k, ret, nnz_q, shift, init, delay_par);
      //////////////////////
      /*
      //int ret = start;
      //int testInd;
      
      int stop = nnz_q;
      int start = ret;
      int elem = k; 
      for (int l = floorf((stop - start)/2.0) + init; l > 1; l -= floorf(l/2.0)){

        testInd = (ret + l < stop - 1) ? ret + l : stop -1;
        ret = (SM_C_q[shift + testInd] <= elem) ? testInd : ret;

      }

      testInd = (ret + 1 < stop - 1) ? ret + 1 : stop - 1;
      ret = (SM_C_q[shift + testInd] <= elem) ? testInd : ret;     

      */
      //////////////////////
      int ind_jk = (SM_C_q[shift + ret] == k) ? ret : -1;
      //if (QId[q] == 678 && glob_pointIds[ptId] == 999692) printf("k = %d, ret = %d, ind_jk = %d , c_tmp = %.6f \n", k, ret, ind_jk, c_tmp); 
      //c_tmp += (ind_jk != -1) ? 0.1 : 0.0;
       
      c_tmp += (ind_jk != -1) ? V_ref[ind0_pt + pos_k] * V_q[ind0_q + ret] : 0.0;
    }
    
    float tmp = -2 * c_tmp + norm_q + Norms_ref[ptId];
    //if (QId[q] == 678 && glob_pointIds[ptId] == 999692) printf("res = %.6f , inner = %.6f, norm_q = %.6f , norm_ref = %.6f \n", tmp, c_tmp, norm_q, Norms_ref[ptId]); 
      
    if (tmp < 1e-8) tmp = 0.0;
    int write_ind = q * ppl + pt; 
    //if (QId[q] == 5 && pt < ppl/2) printf("D[%d] = %.4f at %d , norm_q = %.4f, Norm_ref = %.4f, write_to %d \n", pt, tmp, pt, norm_q, Norms_ref[ptId], write_ind);
    KNN_tmp[write_ind] = tmp;
   
     
  }



}


