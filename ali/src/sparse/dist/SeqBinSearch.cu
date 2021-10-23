#define SM_SIZE_11 2048
#define SM_SIZE_12 4096


__global__ void ComputeNorms(int * R, int* C, float* V, float* Norms){


  int tid = threadIdx.x;
  int pt = blockIdx.y * blockDim.x + blockIdx.x;


  int ind0_i = R[pt];
  int nnz = R[pt+1] - ind0_i;

  float norm_i = 0.0;
  float v;

  for (int n_i = 0; n_i < nnz; n_i++){
    v = V[ind0_i + n_i];
    norm_i += v * v;
  }
  Norms[pt] = norm_i;

}




__device__ int BinSearch(int* arr, int elem, int start, int stop){

  ret = start;

  for (int l = stop - start; l > 1; l -= floorf(l/2.0)){
    testInd = (start + l < stop - 1) ? start + l : stop -1;
    ret = (arr[testInd] <= elem) ? testInd : ret;
  }
  
  testInd = (ret + 1 < stop - 1) ? ret + 1 : stop - 1;
  ret = (arr[testInd] <= elem) ? testInd : ret;
  return ret
}






__global__ ComputeDists(int* R_ref, int* C_ref, float* V_ref, int* R_q, int* C_q, float* V_q, int* leafIds, float* Norms_q, float* Norms_ref, , int const k_nn, float* KNN_tmp, int ppl){
  
  /* 
  R_ref : row pointers to ref points of the selected leaves 
  
  C_ref : col indices of ref points of the selected leaves
  
  V_ref : data of ref points of the selected leaves 
  
  R_q : row pointers to ref points of the selected leaves 
  
  C_q : col indices of ref points of the selected leaves
  
  V_q : data of ref points of the selected leaves 
  
  
  leafIds : array containing the leafId for each query point q
  
  Norms_q : l2 norms of query points
  
  Norms_ref : l2 norms of ref points in all the leaves
  
  k_nn : # neighbors
  
  KNN_tmp : array to store the computed indices 
  
  ppl : # points per leaf
  
  p : number of threads to do parallel intersection 
  
  */

  __shared__ int SM_C_q [SM_SIZE_11];

  int tid = threadIdx.x;

  int q = blockIdx.x;
  int nq = blockDim.x;

  int ind0_q = R_q[q];
  int ind1_q = R_q[q+1];
  int nnz_q = ind1_q - ind0_q;
  float norm_q = Norms_q[q];
   

  for (int n = tid; n < nnz_q; n += blockDim.x) SM_C_q[n] = C[ind0_q + n];
 
  int leafId = leafIds[q];
  int elem = tid;

  
  for (int pt = elem; pt < ppl; pt += blockDim.x){
    
    float c_tmp = 0.0;
    int ptId = leafId * ppl + pt;
    
    ind0_pt = R_ref[ptId];
    nnz_pt = R_ref[ptId+1] - ind0_pt;
    int ret = 0;
  
    for (int pos_k = 0; pos_k < nnz_pt; pos_k++){
      k = C[ind0_pt + pos_k];
      
      ret = BinSearch(SM_C_q, k, ret, nnz_pt);
      int ind_jk = (SM_C_q[ret] == k) ? ret : -1;
      c_tmp += (ind_jk != -1) ? V_ref[ind0_pt + pos_k] * V_q[ind0_q + pos_k]; 
    }
  
    c_tmp = -2 * c_tmp + norm_q + Norms_ref[ptId];
    if (c_tmp < 1e-8) c_tmp = 0.0;
    if (pt < k_nn){    
      KNN_tmp[q * ppl + pt] = c_tmp;
    }
  }



}


