#define SM_SIZE_11 2048
#define SM_SIZE_12 4096
#define SM_SIZE_13 8192
#define SM_SIZE_HALF 6144

#include "GuidedBinSearch.h"


__device__ int BinSearch(int* arr, int elem, int start, int stop, int shift_bs){

  int ret = start;
  int testInd;

  for (int l = stop - start; l > 1; l -= floorf(l/2.0)){

    testInd = (ret + l < stop - 1) ? ret + l : stop -1;
    ret = (arr[shift_bs + testInd] <= elem) ? testInd : ret;

  }
  
  testInd = (ret + 1 < stop - 1) ? ret + 1 : stop - 1;
  ret = (arr[shift_bs + testInd] <= elem) ? testInd : ret;
  
  return ret;
}


__global__ void ComputeSearchGuide(int* R_q, int* C_q, int size, int* SearchInd){


  __shared__ int SM_C_q [SM_SIZE_11];

  int tid = threadIdx.x;

  int q = blockIdx.x;

  int ind0_q = R_q[q];
  int ind1_q = R_q[q+1];
  int nnz_q = ind1_q - ind0_q;

  for (int n = tid; n < nnz_q; n += blockDim.x) SM_C_q[n] = C_q[ind0_q + n];
  
  __syncthreads();
   
  int elem = tid*size;
  int ret = BinSearch(SM_C_q, elem, 0, nnz_q, 0);

  SearchInd[q * blockDim.x + tid] = ret;
  
}


__global__ void ComputeDists_guided(int* R_ref, int* C_ref, float* V_ref, int* R_q, int* C_q, float* V_q, int* local_leafIds, float* Norms_q, float* Norms_ref, int const k_nn, float* KNN_tmp, int const ppl, int* SearchInd, int const size, int const d, int* QId, int const numqsearch){
  

  __shared__ int SM_C_q [SM_SIZE_HALF];
  __shared__ int SM_SearchInd [SM_SIZE_HALF];

  int tid = threadIdx.x;

  int q = blockIdx.x;

  int ind0_q = R_q[q];
  int ind1_q = R_q[q+1];
  int nnz_q = ind1_q - ind0_q;
  float norm_q = Norms_q[q];

  //avoid bank conflict 

  int num_copies_bs = SM_SIZE_HALF / nnz_q;
  int batchsize_bs = blockDim.x / num_copies_bs; 
  if (batchsize_bs * num_copies_bs < blockDim.x) batchsize_bs += 1;
  int batch_bs = tid / batchsize_bs;
  int loop_shift_bs = (blockDim.x < (batch_bs+1) * batchsize_bs) ? blockDim.x - batch_bs * batchsize_bs : batchsize_bs;
  int shift_bs = batch_bs * nnz_q;
 
  
  int num_copies_qs = SM_SIZE_HALF / numqsearch;
  int batchsize_qs = blockDim.x / num_copies_qs; 
  if (batchsize_qs * num_copies_qs < blockDim.x) batchsize_qs += 1;
  int batch_qs = tid / batchsize_qs;
  int loop_shift_qs = (blockDim.x < (batch_qs+1) * batchsize_qs) ? blockDim.x - batch_qs * batchsize_qs : batchsize_qs;
  int shift_qs = batch_qs * numqsearch;
 


  for (int n = tid - batch_bs * batchsize_bs; n < nnz_q; n += loop_shift_bs) SM_C_q[shift_bs + n] = C_q[ind0_q + n];
  for (int n = tid - batch_qs * batchsize_qs; n < numqsearch; n += loop_shift_qs) SM_SearchInd[shift_qs + n] = SearchInd[q * numqsearch + n];
  
  
  __syncthreads(); 


  int leafId = local_leafIds[q];
  int nq = gridDim.x;
   
  for (int pt = tid; pt < ppl; pt += blockDim.x){
    
    float c_tmp = 0.0; 
    int ptId = leafId * ppl + pt;
    
    int ind0_pt = R_ref[ptId];
    int nnz_pt = R_ref[ptId+1] - ind0_pt;
    int ret = 0;
    for (int pos_k = 0; pos_k < nnz_pt; pos_k++){
      int k = C_ref[ind0_pt + pos_k];
      int b = k / size;
      
      int start = (b < numqsearch) ? SM_SearchInd[shift_qs + b] : nnz_q-1;
      int stop = (b+1 < numqsearch) ? SM_SearchInd[shift_qs + b+1] : nnz_q;
      
      int search_start = (start > ret) ? start : ret;
      bool search = true;
      
     
     
     if (SM_C_q[shift_bs + search_start] == k){
        ret = search_start;
        search = false;
      } else if (stop < nnz_q) {
        if (SM_C_q[shift_bs + stop] == k) {
          ret = stop;
          search = false;
        }
      }
      
      ret = (stop > search_start && search==true) ? BinSearch(SM_C_q, k, search_start, stop, shift_bs) : ret; 

      int ind_jk = (SM_C_q[shift_bs + ret] == k) ? ret : -1;
       
      c_tmp += (ind_jk != -1) ? V_ref[ind0_pt + pos_k] * V_q[ind0_q + ret] : 0.0;
  
    }
    float tmp = -2 * c_tmp + norm_q + Norms_ref[ptId];
    //if (q == 0 && pt < 10) printf("cuda p = %d, norms = %.4f, inner = %.4f \n", pt, norm_q + Norms_ref[ptId], c_tmp);  
    if (tmp < 1e-8) tmp = 0.0;
     
    int write_ind = q * ppl + pt; 
    KNN_tmp[write_ind] = tmp;
    
  }
   



}


