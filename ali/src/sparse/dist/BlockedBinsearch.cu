#define SM_SIZE_11 2048
#define SM_SIZE_12 4096



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




__global__ ComputeDists(int* R_ref, int* C_ref, float* V_ref, int* R_q, int* C_q, float* V_q, int* leafIds, float* Norms_q, float* Norms_ref, , int const k_nn, float* KNN_tmp, int ppl, int const nPar){
  
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
  __shared__ int SM_Binsearch_starts[SM_SIZE_12];

  int tid = threadIdx.x;

  int q = blockIdx.x;

  int ind0_q = R_q[q];
  int ind1_q = R_q[q+1];
  int nnz_q = ind1_q - ind0_q;
  
  for (int n = tid; n < nnz_q; n += blockDim.x) SM_C_q[n] = C[ind0_q + n];
  __syncthreads;
 
  int leafId = leafIds[q];
  int elem = tid/p;
  int s = tid - pt * p;

  for (int pt = elem; pt < ppl; pt += blockDim.x){

    ind0_pt = R_ref[leafId + pt];
    nnz_pt = R_ref[leafId + pt] - ind0_pt;
    int b = (nPar > 2) ? nnz_pt / nPar : nnz_pt-1;
    int ind = s * b;
    int start = C_ref[ind0_pt];
    int start_part = BinSearch(SM_C_q, start, 0, nnz_q);
    SM_BinSearch_starts[tid] = start_part;
    __syncthreads();
    int stop_part = (s == nPar - 1) ? nnz_pt : SM_BinSearch_starts[tid+1];
      
 
  } 
  

}







