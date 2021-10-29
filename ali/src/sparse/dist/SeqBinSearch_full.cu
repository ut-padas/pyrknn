#define SM_SIZE_11 2048
#define SM_SIZE_12 4096

#include "SeqBinSearch.h"


__device__ int BinSearch(int* arr, int elem, int start, int stop){

  int ret = start;
  int testInd;

  for (int l = stop - start; l > 1; l -= floorf(l/2.0)){

    testInd = (ret + l < stop - 1) ? ret + l : stop -1;
    ret = (arr[testInd] <= elem) ? testInd : ret;

  }
  
  testInd = (ret + 1 < stop - 1) ? ret + 1 : stop - 1;
  ret = (arr[testInd] <= elem) ? testInd : ret;
  
  return ret;
}


__global__ void ComputeDists_seq(int* R_ref, int* C_ref, float* V_ref, int* R_q, int* C_q, float* V_q, int* leafIds, float* Norms_q, float* Norms_ref, int const k_nn, float* KNN_tmp, int const ppl, int const d, int* QId){
  

  __shared__ int SM_C_q [SM_SIZE_11];

  int tid = threadIdx.x;

  int q = blockIdx.x;

  int ind0_q = R_q[q];
  int ind1_q = R_q[q+1];
  int nnz_q = ind1_q - ind0_q;
  float norm_q = Norms_q[q];
   

  for (int n = tid; n < nnz_q; n += blockDim.x) SM_C_q[n] = C_q[ind0_q + n];
  __syncthreads(); 


  int leafId = leafIds[q];
  int nq = gridDim.x;
  float c_tmp = 0.0; 
  
  for (int pt = tid; pt < ppl; pt += blockDim.x){
    
    c_tmp = 0.0;
    int ptId = leafId * ppl + pt;
    
    int ind0_pt = R_ref[ptId];
    int nnz_pt = R_ref[ptId+1] - ind0_pt;
    int ret = 0;
  
    for (int pos_k = 0; pos_k < nnz_pt; pos_k++){
      int k = C_ref[ind0_pt + pos_k];
      
      ret = BinSearch(SM_C_q, k, ret, nnz_q);
      int ind_jk = (SM_C_q[ret] == k) ? ret : -1;

      c_tmp += (ind_jk != -1) ? V_ref[ind0_pt + pos_k] * V_q[ind0_q + ret] : 0.0;
    }
    c_tmp = -2 * c_tmp + norm_q + Norms_ref[ptId];
    if (c_tmp < 1e-8) c_tmp = 0.0;
    int write_ind = q * ppl + pt; 
    KNN_tmp[write_ind] = c_tmp;
    //if (q == 0 && pt < 100) printf("D[%d] = %.4f , write at %d \n", pt, c_tmp, write_ind); 
     
  }
  



}


