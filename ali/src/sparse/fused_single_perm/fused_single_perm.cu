#define SM_SIZE_10 1024
#define SM_SIZE_11 2048
#define SM_SIZE_12 4096
#define SM_SIZE_13 8192


#include "queryleafknn.h"


__device__ int BinSearch(int* arr, int elem, int start, int stop, int shift){

  int ret = start;
  int testInd;

  for (int l = floorf((stop - start)/2.0); l > 1; l -= floorf(l/2.0)){

    testInd = (ret + l < stop - 1) ? ret + l : stop -1;
    ret = (arr[shift + testInd] <= elem) ? testInd : ret;

  }
  
  testInd = (ret + 1 < stop - 1) ? ret + 1 : stop - 1;
  ret = (arr[shift + testInd] <= elem) ? testInd : ret;
  
  return ret;
}


__global__ void FusedLeafSeqSearch(int* R_ref, int* C_ref, float* V_ref, int* R_q, int* C_q, float* V_q, int* local_leafIds, float* Norms_q, float* Norms_ref, int const k_nn, int const ppl, int* gid_pointIds, float* NDist, int* NId, int steps, int* SortInd, int* StepLen, int* StepStart, int* tidIdMap, int* tidSortDir, int size_batch_iter, int N, int* QId){
  

  __shared__ int SM_C_q [SM_SIZE_12];
  __shared__ float SM_Dist [SM_SIZE_12];
  __shared__ int SM_Id [SM_SIZE_12];

  int tid = threadIdx.x;

  int q = blockIdx.x;

  int ind0_q = R_q[q];
  int ind1_q = R_q[q+1];
  int nnz_q = ind1_q - ind0_q;
  float norm_q = Norms_q[q];
  
  // avoid bank conflict
  /*
  int num_copies = SM_SIZE_12/nnz_q;
  int batchsize = blockDim.x / num_copies;
  if (batchsize * num_copies < blockDim.x) batchsize += 1; 
  int batch = tid / batchsize;
  int loop_shift = (blockDim.x < (batch+1) * batchsize) ? blockDim.x - batch * batchsize : batchsize;
  int shift = batch * nnz_q; 
  */
  int shift = 0;

  for (int n = tid; n < SM_SIZE_12; n += blockDim.x){
    SM_Dist[n] = 1e30;
    SM_Id[n] = -1;
  }

  for (int n = tid; n < nnz_q; n += blockDim.x) SM_C_q[n] = C_q[ind0_q + n];
  //for (int n = tid - batch * batchsize; n < nnz_q; n += loop_shift) SM_C_q[shift + n] = C_q[ind0_q + n];
  for (int n = tid; n < k_nn; n += blockDim.x) {
    SM_Dist[n] = NDist[QId[q] * k_nn + n];
    SM_Id[n] = NId[QId[q] * k_nn + n];
  }
  
  __syncthreads(); 
  
  int nq = gridDim.x;
  float c_tmp;
  int leafId = local_leafIds[q];

	for (int pt = tid; pt < ppl; pt += blockDim.x){
		
		c_tmp = 0.0;
		int ptId = leafId * ppl + pt;
		
		int ind0_pt = R_ref[ptId];
		int nnz_pt = R_ref[ptId+1] - ind0_pt;
		int ret = 0;
	  
		for (int pos_k = 0; pos_k < nnz_pt; pos_k++){
			
			int k = C_ref[ind0_pt + pos_k];
						
			ret = BinSearch(SM_C_q, k, ret, nnz_q, shift);
			//////////////////////
			
			//////////////////////
			int ind_jk = (SM_C_q[shift + ret] == k) ? ret : -1;
			//c_tmp += (ind_jk != -1) ? 0.1 : 0.0;
			 
			c_tmp += (ind_jk != -1) ? V_ref[ind0_pt + pos_k] * V_q[ind0_q + ret] : 0.0;
		}
    
		float tmp = -2 * c_tmp + norm_q + Norms_ref[ptId];
	
		if (tmp < 1e-8) tmp = 0.0;
		
		SM_Dist[tid + k_nn] = tmp;
		SM_Id[tid + k_nn] = gid_pointIds[ptId];
    //if (QId[q] == 5 && tid < 2*k_nn) printf("D[%d] = %.4f at %d \n", tid, SM_Dist[tid], SM_Id[tid]); 
		__syncthreads();
	 
		// check duplicates
		int index = gid_pointIds[ptId];
		for (int ind_check = 0; ind_check < k_nn; ind_check++){
			if (index == SM_Id[ind_check]){
				SM_Id[tid + k_nn] = -1;
				SM_Dist[tid + k_nn] = 1e30;
				break;
			}
		}
		__syncthreads();
    //if (QId[q] == 5 && tid < 2*k_nn) printf("D[%d] = %.4f at %d \n", tid, SM_Dist[tid], SM_Id[tid]); 
		
		__syncthreads();
		int shift = tid * steps;
		int blocksize = (k_nn + blockDim.x)/2.0;
		float tmp_f;
		int tmp_i;
		int check = pt / blockDim.x;

		for (int s = 1; s < steps+1; s++){
			int diff = steps - s;
			int startloc = (tid < blocksize) ? StepStart[shift + diff] : 1;
			int arr_len = (tid < blocksize) ? StepLen[shift +diff] : 1;
			int tid_new = (tid < blocksize) ? tidIdMap[shift +diff] : 1;
			int dir = (tid < blocksize) ? tidSortDir[shift +diff] : 1;
			for (int sl = s-1; sl > -1; sl--){

				int l = 1 << sl;
				int b = (tid_new+0.1) / l;
				int r = tid_new - b * l;
				int j = b * 2 * l + r;

				int ixj = j ^ l;
				j += startloc;
				ixj += startloc;
				bool cond = (dir == -1);
				bool gen_cond = (ixj - startloc < arr_len && arr_len != 1);
				if (tid < blocksize){
					if (gen_cond){
						if (cond){
							if (SM_Dist[j] > SM_Dist[ixj]){

								tmp_f = SM_Dist[j];
								SM_Dist[j] = SM_Dist[ixj];
								SM_Dist[ixj] = tmp_f;

								tmp_i = SM_Id[j];
								SM_Id[j] = SM_Id[ixj];
								SM_Id[ixj] = tmp_i;
							}
						} else {
							if (SM_Dist[j] < SM_Dist[ixj]){
								tmp_f = SM_Dist[j];
								SM_Dist[j] = SM_Dist[ixj];
								SM_Dist[ixj] = tmp_f;

								tmp_i = SM_Id[j];
								SM_Id[j] = SM_Id[ixj];
								SM_Id[ixj] = tmp_i;
							}
						}
					}
				}
				__syncthreads();
			}
		}
		__syncthreads();
    //if (QId[q] == 5 && tid < 2*k_nn) printf("Sorted D[%d] = %.4f at %d \n", tid, SM_Dist[tid], SM_Id[tid]); 
		//__syncthreads();
     
  } 

  if (tid < k_nn){
    int write_ind = QId[q] * k_nn + tid;
    //int write_ind = q * k_nn + tid;
    NDist[write_ind] = SM_Dist[tid];
    NId[write_ind] = SM_Id[tid];
  }

}


