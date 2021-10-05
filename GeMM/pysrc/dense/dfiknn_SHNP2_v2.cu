
#define SM_SIZE_1 1024
#define SM_SIZE_2 2048
#define SM_SIZE_SORT 8192

#include "dfiknn.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <cublas_v2.h>




static const char *cudaGetErrorEnum(cublasStatus_t error) {
    switch (error) {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";

        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";

        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";

        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";

        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";

        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";

        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";

        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";
    }
    return "<unknown>";
}


#define CHECK_CUBLAS(ans) { cublasAssert((ans), __FILE__, __LINE__); }
inline void cublasAssert(cublasStatus_t code, const char *file, int line, bool abort=true) {
   if (code != CUBLAS_STATUS_SUCCESS) {
      fprintf(stderr,"CUBLAS assert: %s %s %d\n", cudaGetErrorEnum(code), file, line);
      if (abort) exit(code);
   }
}



__global__ void ComputeNorms(float* data, int* G_Id, float* Norms, int ppl, int dim, int M) {

  //int row = threadIdx.x + blockIdx.x * blockDim.x;
  int ind = threadIdx.x;
  int leafId_g = blockIdx.z * gridDim.y + blockIdx.y;
  for (int row = ind; row < ppl; row += blockDim.x){
    int g_rowId = leafId_g * ppl + row;
    //changed
  
    int g_Id = g_rowId; 
    int ind0_i = g_Id * dim;
 
    float norm_i = 0.0;
   
    for (int n_i = 0; n_i < dim; n_i += 1) {
      norm_i += data[ind0_i + n_i] * data[ind0_i + n_i];
    }
    Norms[g_Id] = norm_i;
  }
}



__global__ void PrecompMergeNP2(int* SortInd, int* StepLen, int* StepStart, int* tidIdMap, int* tidSortDir, int steps){


  int tid = threadIdx.x;
  int size_sort = 2 * blockDim.x;
  
  int tid1 = 2 * tid + 1;
  int tid2 = tid1 + 1;

  __shared__ int sortind[SM_SIZE_1];
  //if (size_sort == 64) printf("steps = %d \n", steps);

  //SortInd[tid] = 0;
  sortind[tid] = 0;
 
  int shift = tid * steps;
 
  StepLen[shift + 0] = size_sort;
  StepStart[shift + 0] = 0;
  int g_tmp = size_sort;
  int ind = 0;
  int len = g_tmp;
  int direction = -1;
  tidSortDir[shift +0] = direction;
  tidIdMap[shift +0] = tid;
  int loc; 
  for (int s = 1; s < steps; s++){
    g_tmp /= 2;
    
    ind = ind << 1;
      
    if (min(tid1, tid2) > g_tmp){
      ind++;
      tid1 -= g_tmp;
      tid2 -= g_tmp;
      loc = g_tmp + StepStart[shift + s-1];
      len -= g_tmp;
      g_tmp = len;
      
    } else {
      tid2 = (tid2 > g_tmp) ? g_tmp : tid2;
      tid1 = (tid1 > g_tmp) ? g_tmp : tid1;
      loc = StepStart[shift + s-1];
      direction *= -1;
      len = g_tmp;
    }
    StepLen[shift +s] = len;
    StepStart[shift +s] = loc;
    tidSortDir[shift +s] = direction;
     
  }
  //SortInd[tid] = ind;
  sortind[tid] = ind;
  SortInd[tid] = ind;
  tidIdMap[shift + 0] = tid;
  __syncthreads();
  
  //tid_sortdir[0] = -1;
  
  for (int s = 1; s < steps; s++){
    int diff = steps - s - 1;
    int tmp = ind >> diff;
    int testInd = tid;
    int tmp2 = sortind[testInd] >> diff;
    
    int pos = -1;
    while (testInd > -1){
      tmp2 = sortind[testInd] >> diff;
      if (tmp2 == tmp){ 
        pos++; 
      } else{
        break;
      }
      testInd--;
    }
    tidIdMap[shift +s] = pos;
  }
  
 

}


__global__ void MergeHorizNP2(float* KNN, int* KNN_Id, float* Norms, int k_nn, int ppl, int blockInd, float* d_temp_knn, int* G_Id, bool init, int bl, int sizebleaves, int partsize, int steps, int* SortInd, int* step_len, int* step_start, int* tid_idmap, int* tid_sortdir){

   
  __shared__ float SM_dist[SM_SIZE_2];
  __shared__ int SM_Id[SM_SIZE_2];

  int tid = threadIdx.x;
  int row_l = blockIdx.x;
  int leafId_local = blockIdx.z * gridDim.y + blockIdx.y;
  int leafId_g = bl * sizebleaves + leafId_local;
  

  int size_part2 = ppl - partsize * (blockInd + 1);
  int size_part = ppl - partsize * (blockInd);
  int size_sort = 2 * blockDim.x;
  int shift = tid * steps;
  
  int rowId_leaf = partsize * blockInd + row_l;
  
  for (int n=tid; n < SM_SIZE_2; n += blockDim.x){
    SM_dist[n] = 1e30; 
    SM_Id[n] = -1;
  }
  
  float tmp_f;
  int tmp_i;
   
  //int num_batches = size_part / (size_sort - k_nn);
  int num_batches = size_part2 / (size_sort - k_nn);
  if (size_part2 > 0 && num_batches == 0) num_batches += 1; 
  
  float norm_i = Norms[leafId_g * ppl + rowId_leaf];

  
  for (int col_batch = 0; col_batch < num_batches; col_batch++){
    for (int j_tmp = tid; j_tmp < size_sort; j_tmp += blockDim.x){
      
      int colId_leaf = (col_batch == 0) ? partsize * (blockInd + 1) + j_tmp - k_nn: partsize * blockInd + col_batch * (size_sort - k_nn) + j_tmp;
      
      if (col_batch == 0 && j_tmp < k_nn){
        
        int ind_pt = G_Id[leafId_g * ppl + rowId_leaf];
        int ind_read = ind_pt * k_nn + j_tmp;
        SM_Id[j_tmp] = KNN_Id[ind_read];
        SM_dist[j_tmp] = KNN[ind_read];
      } else if (colId_leaf < ppl && j_tmp >= k_nn){

        int size_tmp = size_part - partsize;
        int ind_tmp = leafId_local * partsize * size_tmp + row_l + (colId_leaf - (partsize) * (blockInd+1)) * partsize;
        int g_colId_J = leafId_g * ppl + colId_leaf;
        
        tmp_f = (j_tmp < k_nn + size_part2) ? -2 * d_temp_knn[ind_tmp] + norm_i + Norms[leafId_g * ppl + colId_leaf] : 1e30;
        SM_dist[j_tmp] = (tmp_f > 0) ? tmp_f : 0.0;
        SM_Id[j_tmp] = G_Id[g_colId_J];
        //if (rowId_leaf == 512 && leafId_g == 0 && colId_leaf < 700) printf("Covering Col = %d \n", colId_leaf);
      }
      //if (rowId_leaf == 22 && leafId_g == 1) printf("D[%d] = %.4f at %d \n", j_tmp, SM_dist[j_tmp], SM_Id[j_tmp]);   
 
    }

    __syncthreads();
     
    for (int j_tmp = tid; j_tmp < size_sort; j_tmp += blockDim.x) {

      if (j_tmp >= k_nn){
        int index = SM_Id[j_tmp];
        for (int ind_check = 0; ind_check < k_nn; ind_check++){
          if (index == SM_Id[ind_check]){
            SM_Id[j_tmp] = -1;
            SM_dist[j_tmp] = 1e25;
            break;
          }
        }
      }

    //if (rowId_leaf == 22 && leafId_g == 1) printf("D[%d] = %.4f at %d \n", j_tmp, SM_dist[j_tmp], SM_Id[j_tmp]);   
    }
    
    __syncthreads();
    //if (rowId_leaf == 0) printf("D[%d] = %.4f at %d \n", tid, SM_dist[tid], SM_Id[tid]);   
    /* 
    if (rowId_leaf == 0){
      for (int s=0; s < steps; s++) printf("tid = %d, arr_len[%d] = %d \n", tid, s, step_len[s]);
    }
    */
		for (int s = 1; s < steps+1; s++){
			int diff = steps - s;
			int startloc = step_start[shift + diff];
			int arr_len = step_len[shift +diff];
			int tid_new = tid_idmap[shift +diff];
			int dir = tid_sortdir[shift +diff];
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
        //if (rowId_leaf ==22 && leafId_g == 0) printf("s = %d, l=%d,tid = %d, tid_new = %d, b = %d, r = %d, j = %d, ixj = %d, startloc = %d, arr_len = %d, cond = %d \n", s, l, tid, tid_new, b, r, j, ixj, startloc,arr_len, cond);   
				if (gen_cond){
					if (cond){
						if (SM_dist[j] > SM_dist[ixj]){

							tmp_f = SM_dist[j];
							SM_dist[j] = SM_dist[ixj];
							SM_dist[ixj] = tmp_f;
		
							tmp_i = SM_Id[j];
							SM_Id[j] = SM_Id[ixj];
							SM_Id[ixj] = tmp_i;
						}
					} else {
						if (SM_dist[j] < SM_dist[ixj]){
							tmp_f = SM_dist[j];
							SM_dist[j] = SM_dist[ixj];
							SM_dist[ixj] = tmp_f;
		
							tmp_i = SM_Id[j];
							SM_Id[j] = SM_Id[ixj];
							SM_Id[ixj] = tmp_i;
						}
            
          }
        }
        __syncthreads();
      }
    } 

  }
  //if (rowId_leaf == 0) printf("s = %d, l=%d,tid = %d,  j = %d, ixj = %d, startloc = %d, arr_len = %d, cond = %d \n", s, l, tid, j, ixj, startloc,arr_len, cond);   
  //if (rowId_leaf == 22 && leafId_g == 1) printf("sorted D[%d] = %.4f at %d \n", tid, SM_dist[tid], SM_Id[tid]);   
  //if (rowId_leaf == 0 && leafId_g == 0) printf("sorted D[%d] = %.4f at %d \n", tid, SM_dist[tid], SM_Id[tid]);   
  for (int j_tmp = tid; j_tmp < k_nn; j_tmp += blockDim.x){ 
    //if (j_tmp < k_nn){
      int ind_pt = leafId_g * ppl + rowId_leaf;
      int write_ind = G_Id[ind_pt] * k_nn + j_tmp;
      KNN[write_ind] = SM_dist[j_tmp];
      KNN_Id[write_ind] = SM_Id[j_tmp];
    //}
  } 
}


__global__ void ComputeTriDists_last(float* data, int* G_Id, float* Norms , int k_nn, float* KNN_dist_tmp, int ppl, int rem_len , int blockId, int bl, int sizebleaves, int partsize, int dim) {




  int ind = threadIdx.x;
  //int leaf_id_g = blockIdx.z * blockDim.y + blockIdx.y;
  int leafId_local = blockIdx.z * gridDim.y + blockIdx.y;
  int leafId_g = bl * sizebleaves + leafId_local;
  int block = blockId;



  int size_block = rem_len * (rem_len + 1) /2;


  for (int elem = ind; elem < size_block; elem += blockDim.x){

    float tmp = -8 * elem + 4 * rem_len * (rem_len+1) - 7;
    int rowId = sqrt(tmp)/2.0 - 0.5;
    rowId = rem_len - 1 - rowId;
    int colId = elem + rowId - rem_len * (rem_len + 1) / 2 + (rem_len - rowId) * ((rem_len - rowId) + 1)/2;

    float c_tmp = 0.0;
    //if (block * k_nn + rowId < ppl && block * k_nn + colId < ppl){
    if (block * partsize + rowId < ppl && block * partsize + colId < ppl){

    //int g_rowId = leaf_id_g * ppl + block * k_nn + rowId;
    //int g_colId = leaf_id_g * ppl + block * k_nn + colId;
    //int g_rowId = leafId_g * ppl + block * k_nn + rowId;
    //int g_colId = leafId_g * ppl + block * k_nn + colId;
    int g_rowId = leafId_g * ppl + block * partsize + rowId;
    int g_colId = leafId_g * ppl + block * partsize + colId;

    //changed
    int perm_i = g_rowId;
    int perm_j = g_colId;

    int ind0_i = perm_i * dim;

    int ind0_j = perm_j * dim;


		for (int pos_k = 0; pos_k < dim; pos_k++){
			c_tmp += data[ind0_j + pos_k] * data[ind0_i + pos_k];

    }
    } else {
      c_tmp = -1e30;
    }

    /*
    int gid_pt = leafId_local * ppl + block * partsize + rowId;
    int gid_pt_T = leafId_local * ppl + block * partsize + colId;
    int ind_knn = gid_pt * partsize + colId;
    int ind_knn_T = gid_pt_T * partsize + rowId;
    */
    int gid_pt = leafId_local * ppl * partsize + block * partsize * partsize + rowId * partsize;
    int gid_pt_T = leafId_local * ppl * partsize + block * partsize * partsize + colId * partsize;
    int ind_knn = gid_pt + colId;
    int ind_knn_T = gid_pt_T + rowId;



    KNN_dist_tmp[ind_knn] = c_tmp;
    if (colId > rowId) KNN_dist_tmp[ind_knn_T] = c_tmp;
    //if (leafId_g * ppl + block * partsize + rowId == 2339) printf("write for last D[%d] = %.4f write at %d \n", colId, c_tmp, ind_knn); 
    //if (leafId_g * ppl + block * partsize + colId == 2339) printf("write for last D[%d] = %.4f write at %d \n", rowId, c_tmp, ind_knn_T); 
    /*
    for (int row_tmp = 0; row_tmp<rem_len; row_tmp++){
      for (int q = ind + rem_len; q < partsize; q += blockDim.x){
        //gid_pt = leafId_local * ppl + block * k_nn + row_tmp;
        //ind_knn = gid_pt * k_nn + q;
        gid_pt = leafId_local * ppl * partsize + block * partsize * partsize + row_tmp;
        ind_knn = gid_pt + q * partsize;
        
        if (ind_knn < leafId_local * ppl * partsize) KNN_dist_tmp[ind_knn] = -1e30;
        //if (leafId_g * ppl + block * partsize + row_tmp == 2339) printf("write for last D[%d] = %.4f , write at %d \n", q, KNN_dist_tmp[ind_knn], ind_knn);
      }
    }
    //if (leafId_g == 1) printf("rows = %d \n", );
    */ 
  }

}


__global__ void MergeVer_v2(float* KNN, int* KNN_Id, float* Norms, int k_nn, int ppl, int blockInd, float* d_temp_knn, int* G_Id, bool init, int bl, int sizebleaves, int partsize){

  __shared__ float SM_dist[SM_SIZE_1];
  __shared__ int SM_Id[SM_SIZE_1];

  int j = threadIdx.x;
  float tmp_f;
  int tmp_i;

  int col = blockIdx.x;
  int leafId_local = blockIdx.z* gridDim.y + blockIdx.y;
  int leafId_g = bl * sizebleaves + leafId_local;
  
  int colId_leaf = (init) ? col : col + partsize * (blockInd + 1);
  int size_part = (init) ? ppl : ppl - (blockInd + 1) * partsize;
  

  float norm_i = Norms[leafId_g * ppl + colId_leaf];

	int ind_pt_knn = leafId_g * ppl + colId_leaf;
	int ind_pt_knn_g = G_Id[ind_pt_knn];

  for (int ind = j; ind < 2 * partsize; ind += blockDim.x){
    SM_dist[ind] = 1e30;
    SM_Id[ind] = -1;
  }
  __syncthreads();

  
  for (int j_tmp = j; j_tmp < partsize; j_tmp += blockDim.x){
		
    //int ind_tmp = (init) ? leafId_local * ppl * partsize + col * partsize + j_tmp : leafId_local * partsize * size_part + j_tmp + col * partsize;
    int ind_tmp = (init) ? leafId_local * ppl * partsize + col * partsize + j_tmp : leafId_local * partsize * size_part + j_tmp + col * partsize;
	  
		int block = col / partsize;
		int M = ppl * gridDim.y * gridDim.z; 

    int rowId_g = (init) ? leafId_g * ppl + block * partsize + j_tmp : leafId_g * ppl + partsize * blockInd + j_tmp;

    int Max_blocks = ppl/partsize;
    int rem_len = ppl - Max_blocks * partsize;

		SM_Id[j_tmp] = (rowId_g < M) ? G_Id[rowId_g] : -1;
    tmp_f = (rowId_g < M) ? -2 * d_temp_knn[ind_tmp] + norm_i + Norms[rowId_g] : 1e30;
    //tmp_f = d_temp_knn[ind_tmp] + norm_i + Norms[rowId_g];
    
    if (init){
      tmp_f = (block * partsize + j_tmp < ppl) ? tmp_f : 1e30;
      tmp_f = (Max_blocks == block && j_tmp >= rem_len) ? 1e30 : tmp_f;
    }
     
  	SM_dist[j_tmp] = (tmp_f > 0.0) ? tmp_f : 0.0;
 
		int ind_knn = ind_pt_knn_g * k_nn + j_tmp;
		if (j_tmp < k_nn){ 
			SM_dist[j_tmp + partsize] = KNN[ind_knn];
			SM_Id[j_tmp + partsize] = KNN_Id[ind_knn];
		} else {
      SM_dist[j_tmp + partsize] = 1e30;
      SM_Id[j_tmp + partsize] = -1;
    }
    
	  __syncthreads();
    //if (leafId_g == 1 && colId_leaf == 0) printf("v tmp D[%d] = %.4f at %d \n", j_tmp, SM_dist[j_tmp], SM_Id[j_tmp]);
    //if (leafId_g == 1 && colId_leaf == 0) printf("v knn D[%d] = %.4f at %d \n", j_tmp+partsize, SM_dist[j_tmp+partsize], SM_Id[j_tmp+partsize]);
    //if (ind_pt_knn == 1170 && init == 1) printf("tmp D[%d] = %.4f at %d  , reading from %d \n", j_tmp, SM_dist[j_tmp], SM_Id[j_tmp], ind_tmp);
    //if (ind_pt_knn == 1170 && init == 1) printf("knn D[%d] = %.4f at %d , reading from %d \n", j_tmp+partsize, SM_dist[j_tmp], SM_Id[j_tmp+partsize], ind_tmp);
    //if (ind_pt_knn == 1170 && init == 1) printf("tmp D[%d] = %.4f at %d  , reading from %d \n", j_tmp, d_temp_knn[ind_tmp], SM_Id[j_tmp], ind_tmp);
    //if (ind_pt_knn == 1170 && init == 1) printf("knn D[%d] = %.4f at %d , reading from %d \n", j_tmp+partsize, SM_dist[j_tmp + partsize], SM_Id[j_tmp+partsize], ind_tmp);

  }
  
	__syncthreads();
  

  for (int j_tmp = j; j_tmp < partsize; j_tmp += blockDim.x){ 
    int index = SM_Id[j_tmp];
    for (int ind_check = 0; ind_check < k_nn; ind_check++){
      if (index == SM_Id[ind_check + partsize]){
        SM_dist[j_tmp] = 1e30;
        SM_Id[j_tmp] = -1;
        break;
      }
    }
  }
  __syncthreads();



  //int size_sort = 2 * partsize;
    

  for (int g = 2; g <= 2 * blockDim.x; g *= 2){
    for (int l = g/2; l > 0; l /= 2){
      int tid = j;
      int ixj = tid ^ l;
      if (ixj < tid){
        tid += blockDim.x;
        ixj = tid ^ l;
        if (ixj < tid){
          tid = ixj;
          ixj = tid ^ l;
        }
      }
      
      if ((tid & g) == 0){

        if (SM_dist[tid] > SM_dist[ixj]){
          tmp_f = SM_dist[ixj];
          SM_dist[ixj] = SM_dist[tid];
          SM_dist[tid] = tmp_f;
          tmp_i = SM_Id[ixj];
          SM_Id[ixj] = SM_Id[tid];
          SM_Id[tid] = tmp_i;
        }
      } else {
        if (SM_dist[tid] < SM_dist[ixj]){
          tmp_f = SM_dist[ixj];
          SM_dist[ixj] = SM_dist[tid];
          SM_dist[tid] = tmp_f;
          tmp_i = SM_Id[ixj];
          SM_Id[ixj] = SM_Id[tid];
          SM_Id[tid] = tmp_i;
        }
      }
      __syncthreads();
    }
  }
  //if (leafId_g == 1 && colId_leaf == 0) printf("v Sorted D[%d] = %.4f at %d \n", j, SM_dist[j], SM_Id[j]);
  //if (leafId_g == 1 && colId_leaf == 0) printf("v Sorted D[%d] = %.4f at %d \n", j+partsize, SM_dist[j+partsize], SM_Id[j+partsize]);
  __syncthreads();
  if (j < k_nn){ 
		int ind_pt_knn = leafId_g * ppl + colId_leaf;
		int ind_pt_knn_g = G_Id[ind_pt_knn];
		int ind_knn = ind_pt_knn_g * k_nn + j;
		KNN[ind_knn] = SM_dist[j];
		KNN_Id[ind_knn] = SM_Id[j];
  } 
   

}


void dfi_leafknn(float *d_data, int *d_GId, int M, int leaves, int k, float *d_knn, int *d_knn_Id, int dim){



  float dt_mem, dt_norms, dt_diagdist, dt_diagmerge, dt_mergehoriz, dt_mergever, dt_recdist, dt_tot,dt_tmp;
  
  dt_mem = 0.0;
  dt_norms = 0.0;
  dt_diagdist = 0.0;
  dt_diagmerge = 0.0;
  dt_mergehoriz = 0.0;
  dt_mergever = 0.0;
  dt_recdist = 0.0;
  dt_tot = 0.0;
  dt_tmp = 0.0;
  cudaEvent_t t_begin;
  cudaEvent_t t_end;
  
  cudaEvent_t t0_mem;
  cudaEvent_t t1_mem;

  cudaEvent_t t0_norms;
  cudaEvent_t t1_norms;

  cudaEvent_t t0_diagdist;
  cudaEvent_t t1_diagdist;

  cudaEvent_t t0_diagmerge;
  cudaEvent_t t1_diagmerge;

  cudaEvent_t t0_mergehoriz;
  cudaEvent_t t1_mergehoriz;
  
  cudaEvent_t t0_mergever;
  cudaEvent_t t1_mergever;

  cudaEvent_t t0_recdist;
  cudaEvent_t t1_recdist;


  checkCudaErrors(cudaEventCreate(&t_begin));
  checkCudaErrors(cudaEventCreate(&t_end));

  checkCudaErrors(cudaEventCreate(&t0_mem));
  checkCudaErrors(cudaEventCreate(&t1_mem));

  checkCudaErrors(cudaEventCreate(&t0_norms));
  checkCudaErrors(cudaEventCreate(&t1_norms));

  checkCudaErrors(cudaEventCreate(&t0_diagdist));
  checkCudaErrors(cudaEventCreate(&t1_diagdist));

  checkCudaErrors(cudaEventCreate(&t0_diagmerge));
  checkCudaErrors(cudaEventCreate(&t1_diagmerge));

  checkCudaErrors(cudaEventCreate(&t0_mergehoriz));
  checkCudaErrors(cudaEventCreate(&t1_mergehoriz));

  checkCudaErrors(cudaEventCreate(&t0_mergever));
  checkCudaErrors(cudaEventCreate(&t1_mergever));

  checkCudaErrors(cudaEventCreate(&t0_recdist));
  checkCudaErrors(cudaEventCreate(&t1_recdist));


  checkCudaErrors(cudaEventRecord(t_begin, 0));
  checkCudaErrors(cudaEventSynchronize(t_begin));
  int verbose = 1;
  if (verbose) printf("----------------------------- Start of sfiknn ----------------------------- \n\n");


  //int C_len = M * dim;


  size_t free, total, m1, m2, m3;

  int ppl = M/leaves;


  int partsize = (k > 32) ? k : 32;
  //int partsize = k;

  cudaMemGetInfo(&free, &total);
  if (verbose) printf(" Available Memory : %.4f GB from %.4f \n", free/1e9, total/1e9);
  size_t size_req = sizeof(float) * partsize * M;
  int counter =0;
  bool flag_div = false;
  while (size_req < free && partsize < k && counter < 6) {
    counter++;
    size_req *= 2;
    partsize *= 2;
    flag_div = true;
    //if (verbose) printf("partsize = %d,  free = %.4f , size_req = %.4f,\n", partsize, free/1e9, size_req/1e9);
  }
  if (flag_div) partsize /= 2;
  partsize = (partsize > ppl) ? ppl : partsize;
  partsize = (partsize < k) ? k : partsize;


  int num_blocks_tri = ppl / partsize;
  
  //if (num_blocks_tri * k < ppl) num_blocks_tri += 1;
  //int rem_len = (num_blocks_tri * k < ppl) ? ppl - num_blocks_tri * k : 0;
  int rem_len = (num_blocks_tri * partsize < ppl) ? ppl - num_blocks_tri * partsize : 0;
 


  int t_b = (ppl > SM_SIZE_1) ? SM_SIZE_1 : ppl;
  //float tmp = leaves / 65535;
  int num_splits = 1;
  while (leaves > num_splits * 65535) num_splits *= 2;
  //int num_splits = ceil(tmp);

  int batch_leaves_1 = (leaves > 64000) ? leaves / num_splits : leaves;
  int batch_leaves_2 = (leaves > 64000) ? num_splits : 1;




  int size_tri = partsize;
  int blockDim_tri = size_tri * (size_tri + 1)/2;
  if (blockDim_tri > SM_SIZE_1) blockDim_tri = SM_SIZE_1;

  int size_tri_last = (rem_len > 32) ? 32 : rem_len;
  int blockDim_tri_last = size_tri_last * (size_tri_last + 1)/2;
  if (blockDim_tri_last > SM_SIZE_1) blockDim_tri_last = SM_SIZE_1;

	dim3 BlockDistTri_last(blockDim_tri_last, 1, 1);
	dim3 GridDistTri_last(1, batch_leaves_1, batch_leaves_2);

  dim3 BlockNorm(t_b, 1, 1);
  dim3 GridNorm(1, batch_leaves_1, batch_leaves_2);
  
  dim3 GridMergeHoriz(partsize, batch_leaves_1, batch_leaves_2);

  //int size_v_block_reduced = (k + partsize)/2;
  int size_v_block_reduced = partsize;
  dim3 BlockMergeVer(size_v_block_reduced, 1, 1);
  if (verbose){
  printf("=======================\n");
  printf(" Num points = %d \n", M);
  printf(" pt/leaf = %d \n", ppl);
  printf(" Leaves = %d \n", leaves);
  printf(" K = %d \n", k);
  printf(" PartSize = %d \n", partsize);

  //printf(" dim BlockThreads  Norms = (%d , %d, %d) \n", BlockNorm.x, BlockNorm.y, BlockNorm.z);
  //printf(" dim GridThreads Norms = (%d , %d, %d) \n", GridNorm.x, GridNorm.y, GridNorm.z);
  printf(" dim GridThreads MergeHoriz = (%d , %d, %d) \n", GridMergeHoriz.x, GridMergeHoriz.y, GridMergeHoriz.z);
  printf(" dim BlockMerge MergeVer = (%d , %d, %d) \n", BlockMergeVer.x, BlockMergeVer.y, BlockMergeVer.z);
  }



  //int *d_arr, *d_arr_part;
  float SM_SIZE_2_f = SM_SIZE_2;
  int n_s = log2(SM_SIZE_2_f) *(log2(SM_SIZE_2_f)+1) /2;

  //int copy_size = (ppl) * n_s;
  //float tmp = 2*k;
  //float tmp = 2*k;
  float tmp = 2*partsize;
  int n_s_v = log2(tmp) * (log2(tmp)+1)/2;
  //int copy_size_v = k * n_s;
  //int copy_size_v = (2 * partsize) * n_s_v;



    
  checkCudaErrors(cudaEventRecord(t0_mem, 0));
  checkCudaErrors(cudaEventSynchronize(t0_mem));
    
  cudaMemGetInfo(&free, &total);
  int *SortInd, *StepLen, *StepStart, *tidIdMap, *tidSortDir;
 
  checkCudaErrors(cudaMalloc((void **) &SortInd, sizeof(int) * SM_SIZE_1));
  checkCudaErrors(cudaMalloc((void **) &StepLen, sizeof(int) * 12 * SM_SIZE_1));
  checkCudaErrors(cudaMalloc((void **) &StepStart, sizeof(int) * 12* SM_SIZE_1));
  checkCudaErrors(cudaMalloc((void **) &tidIdMap, sizeof(int) * 12* SM_SIZE_1));
  checkCudaErrors(cudaMalloc((void **) &tidSortDir, sizeof(int) * 12* SM_SIZE_1));

  //checkCudaErrors(cudaMemset(d_arr, 0, sizeof(int) * copy_size));
  //checkCudaErrors(cudaMemset(d_arr_part, 0, sizeof(int) * copy_size));
  //checkCudaErrors(cudaMemset(d_arr_v, 0, sizeof(int) * copy_size_v));
  //checkCudaErrors(cudaMemset(d_arr_part_v, 0, sizeof(int) * copy_size_v));
  cudaMemGetInfo(&m1, &total);



  
  //int size_sort_ver = k + partsize;
  //int size_sort_ver_pow2 = 2*partsize;
  //PrecompSortIds(d_arr_v, d_arr_part_v, size_sort_ver, size_sort_ver_pow2, n_s_v, copy_size_v);



  float *d_temp_knn, *d_Norms;
  checkCudaErrors(cudaMalloc((void **) &d_Norms, sizeof(float) * M));



  cudaMemGetInfo(&m2, &total);
  size_t size_tmp = sizeof(float) * M * partsize;
   if (verbose) printf("Needed %.4f GB got %.4f \n", size_tmp/1e9, m2/1e9);
  //float s1 = size_tmp/1e9;
  //float s2 = m2/1e9;
  float tmp2 = ceil(size_tmp/m2);
  
  int bleaves = (size_tmp > m2) ? log2(tmp2) : 0;
  int numbleaves = 1 << bleaves;
  int sizebleaves = leaves / numbleaves; 
  //printf(" tmp2 = %.4f , bleaves = %d , numbleaves = %d \n", tmp2, bleaves, numbleaves);
   if (verbose){
  printf(" Num BatchLeaves = %d \n", numbleaves);
  printf(" Size BatchLeaves = %d \n", sizebleaves);
  printf("=======================\n");
  }
  
  
  checkCudaErrors(cudaMalloc((void **) &d_temp_knn, sizeof(float) * sizebleaves * ppl * partsize));
  cudaMemGetInfo(&m3, &total);

  checkCudaErrors(cudaEventRecord(t1_mem, 0));
  checkCudaErrors(cudaEventSynchronize(t1_mem));
  checkCudaErrors(cudaEventElapsedTime(&dt_tmp, t0_mem, t1_mem));
	dt_mem = dt_tmp;
  int steps;

	cublasStatus_t status;
	cublasHandle_t handle;

  status = cublasCreate(&handle);
  //int oneInt = 1;
  float oneFloat = 1.0;

  int num_gemms = ppl / partsize;
  //num_gemms *= sizebleaves;
  float zeroFloat = 0.0;

  /*   
  CHECK_CUBLAS( cublasSgemmStridedBatched( handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                      oneInt, oneInt, dim,
                                      &oneFloat, d_data, dim, dim,
                                      d_data, dim, dim, 
                                      &zeroFloat, d_Norms, oneInt, oneInt, M) );
  */

  checkCudaErrors(cudaEventRecord(t0_norms, 0));
  ComputeNorms <<< GridNorm, BlockNorm >>>(d_data, d_GId, d_Norms, ppl, dim, M); 
 
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaEventRecord(t1_norms, 0));
  checkCudaErrors(cudaEventSynchronize(t1_norms));
  checkCudaErrors(cudaEventElapsedTime(&dt_tmp, t0_norms, t1_norms));
	dt_norms = dt_tmp;
  //cudaStream_t streams[2];
  //cudaStreamCreate(&streams[0]);
  //cudaStreamCreate(&streams[1]);
  for (int bl = 0; bl < numbleaves; bl++){

    checkCudaErrors(cudaEventRecord(t0_diagdist, 0));
    for (int l = 0; l < sizebleaves; l++){
  

    CHECK_CUBLAS( cublasSgemmStridedBatched( handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                      partsize, partsize, dim,
                                      &oneFloat, d_data + l * ppl * dim + bl * sizebleaves * ppl * dim, dim, partsize*dim,
                                      d_data + l * ppl * dim + bl * sizebleaves * ppl * dim, dim, partsize*dim, 
                                      &zeroFloat, d_temp_knn + l * ppl * partsize, partsize, partsize*partsize, num_gemms) );
    }

    

		checkCudaErrors(cudaDeviceSynchronize());
    
    
 
    
    if (rem_len > 0){
      //printf("rem len = %d , num_gemms = %d ,\n", rem_len, num_gemms);
      ComputeTriDists_last <<< GridDistTri_last, BlockDistTri_last >>>(d_data, d_GId, d_Norms, k, d_temp_knn, ppl, rem_len, num_gemms, bl, sizebleaves, partsize, dim);
      checkCudaErrors(cudaDeviceSynchronize());

    }
    checkCudaErrors(cudaEventRecord(t1_diagdist, 0));   
    checkCudaErrors(cudaEventSynchronize(t1_diagdist));
    checkCudaErrors(cudaEventElapsedTime(&dt_tmp, t0_diagdist, t1_diagdist));
    dt_diagdist += dt_tmp;
     
   
      
    
 
		int size_v = ppl;
		dim3 GridMergeVer(size_v, batch_leaves_1, batch_leaves_2);
    
		//MergeVer <<< GridMergeVer, BlockMergeVer >>> (d_knn, d_knn_Id, d_Norms, k, ppl, 0, d_temp_knn, d_arr_v, d_arr_part_v, n_s_v, d_GId, true, bl, sizebleaves, partsize);
    checkCudaErrors(cudaEventRecord(t0_diagmerge, 0));
		MergeVer_v2 <<< GridMergeVer, BlockMergeVer >>> (d_knn, d_knn_Id, d_Norms, k, ppl, 0, d_temp_knn, d_GId, true, bl, sizebleaves, partsize);
		//checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaEventRecord(t1_diagmerge, 0));
    checkCudaErrors(cudaEventSynchronize(t1_diagmerge));

    checkCudaErrors(cudaEventElapsedTime(&dt_tmp, t0_diagmerge, t1_diagmerge));
    dt_diagmerge += dt_tmp;
  
		int num_iters = (rem_len > 0) ? num_blocks_tri : num_blocks_tri - 1;
		for (int blockInd = 0; blockInd < num_iters; blockInd++){

			
      int size_part = ppl - (blockInd+1) * partsize + k;
			int size_sort = size_part;

			while (size_sort > SM_SIZE_2) size_sort = ceil((size_sort+k)/2);
			float tmp = size_sort/2.0;
			int blocksize = ceil(tmp);
			float tmp_f = 2 * blocksize;
			int N_pow2 = pow(2, ceil(log2(tmp_f)));
			tmp_f = N_pow2;
			steps = log2(tmp_f);
			

			dim3 BlockMergeHoriz( blocksize, 1, 1);

			int size_v2 = ppl - (blockInd + 1) * partsize;
			dim3 GridMergeVer(size_v2, batch_leaves_1, batch_leaves_2);

      

      int offsetA =  partsize * blockInd + bl * sizebleaves * ppl;
      int offsetB = partsize * (blockInd + 1) + bl * sizebleaves * ppl;
      num_gemms = sizebleaves;
      int sizecolumns = ppl - partsize * (blockInd + 1);
      checkCudaErrors(cudaEventRecord(t0_recdist, 0));

      CHECK_CUBLAS( cublasSgemmStridedBatched( handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                      partsize, sizecolumns, dim,
                                      &oneFloat, d_data + offsetA * dim, dim, ppl * dim,
                                      d_data + offsetB * dim, dim, ppl * dim, 
                                      &zeroFloat, d_temp_knn, partsize, sizecolumns * partsize, num_gemms) );           
    	checkCudaErrors(cudaDeviceSynchronize());
      checkCudaErrors(cudaEventRecord(t1_recdist, 0));
      checkCudaErrors(cudaEventSynchronize(t1_recdist));
      checkCudaErrors(cudaEventElapsedTime(&dt_tmp, t0_recdist, t1_recdist));
      dt_recdist += dt_tmp;


      checkCudaErrors(cudaEventRecord(t0_mergehoriz, 0));
       
      PrecompMergeNP2 <<< 1, blocksize >>> (SortInd, StepLen, StepStart, tidIdMap, tidSortDir, steps);
		  checkCudaErrors(cudaDeviceSynchronize());	
      
      MergeHorizNP2 <<< GridMergeHoriz, BlockMergeHoriz>>> (d_knn, d_knn_Id, d_Norms, k, ppl, blockInd, d_temp_knn, d_GId, false, bl, sizebleaves, partsize, steps, SortInd, StepLen, StepStart, tidIdMap, tidSortDir); 
      checkCudaErrors(cudaEventRecord(t1_mergehoriz, 0));
      checkCudaErrors(cudaEventSynchronize(t1_mergehoriz));
      checkCudaErrors(cudaEventElapsedTime(&dt_tmp, t0_mergehoriz, t1_mergehoriz));
      dt_mergehoriz += dt_tmp;
		
			//MergeVer <<< GridMergeVer, BlockMergeVer , 0, streams[1]>>> (d_knn, d_knn_Id, d_Norms, k, ppl, blockInd, d_temp_knn, d_arr_v, d_arr_part_v, n_s_v, d_GId, false,bl, sizebleaves, partsize);
			//MergeVer_v2 <<< GridMergeVer, BlockMergeVer , 0, streams[1]>>> (d_knn, d_knn_Id, d_Norms, k, ppl, blockInd, d_temp_knn, d_GId, false,bl, sizebleaves, partsize);
      checkCudaErrors(cudaEventRecord(t0_mergever, 0));
			MergeVer_v2 <<< GridMergeVer, BlockMergeVer>>> (d_knn, d_knn_Id, d_Norms, k, ppl, blockInd, d_temp_knn, d_GId, false,bl, sizebleaves, partsize);
			checkCudaErrors(cudaDeviceSynchronize());
      checkCudaErrors(cudaEventRecord(t1_mergever, 0));
      checkCudaErrors(cudaEventSynchronize(t1_mergever));
      checkCudaErrors(cudaEventElapsedTime(&dt_tmp, t0_mergever, t1_mergever));
      dt_mergever += dt_tmp;
    }
  
  }

  checkCudaErrors(cudaEventRecord(t_end, 0));
  checkCudaErrors(cudaEventSynchronize(t_end));
  checkCudaErrors(cudaEventElapsedTime(&dt_tot, t_begin, t_end));
  checkCudaErrors(cudaEventElapsedTime(&dt_tmp, t_begin, t0_mem));



  //checkCudaErrors(cudaStreamDestroy(streams[0]));
  //checkCudaErrors(cudaStreamDestroy(streams[1]));
  checkCudaErrors(cudaFree(d_Norms));
  checkCudaErrors(cudaFree(d_temp_knn));
  //checkCudaErrors(cudaFree(d_arr_part));
  //checkCudaErrors(cudaFree(d_arr));
  
  checkCudaErrors(cudaFree(SortInd));
  checkCudaErrors(cudaFree(StepLen));
  checkCudaErrors(cudaFree(StepStart));
  checkCudaErrors(cudaFree(tidIdMap));
  checkCudaErrors(cudaFree(tidSortDir));
  
  //checkCudaErrors(cudaDeviceSynchronize());
 


 
  checkCudaErrors(cudaEventDestroy(t_begin));
  checkCudaErrors(cudaEventDestroy(t_end));
  checkCudaErrors(cudaEventDestroy(t0_mem));
  checkCudaErrors(cudaEventDestroy(t1_mem));
  checkCudaErrors(cudaEventDestroy(t0_norms));
  checkCudaErrors(cudaEventDestroy(t1_norms));
  checkCudaErrors(cudaEventDestroy(t0_diagdist));
  checkCudaErrors(cudaEventDestroy(t1_diagdist));
  checkCudaErrors(cudaEventDestroy(t0_diagmerge));
  checkCudaErrors(cudaEventDestroy(t1_diagmerge));
  checkCudaErrors(cudaEventDestroy(t0_mergehoriz));
  checkCudaErrors(cudaEventDestroy(t1_mergehoriz));
  checkCudaErrors(cudaEventDestroy(t0_mergever));
  checkCudaErrors(cudaEventDestroy(t1_mergever));
  checkCudaErrors(cudaEventDestroy(t0_recdist));
  checkCudaErrors(cudaEventDestroy(t1_recdist));

  if (verbose){
  printf("--------------- Timings ----------------\n");
  printf(" Initialization = %.4f (%.f %%) \n", dt_tmp/1e3, 100*dt_tmp/dt_tot);
  printf(" Memory allocation = %.4f (%.f %%) \n", dt_mem/1e3, 100*dt_mem/dt_tot);
  printf(" Computing norms = %.4f (%.f %%) \n", dt_norms/1e3, 100*dt_norms/dt_tot);
  printf(" Diagonal dist = %.4f (%.f %%) \n", dt_diagdist/1e3, 100*dt_diagdist/dt_tot);
  printf(" Diagonal merge = %.4f (%.f %%) \n", dt_diagmerge/1e3, 100*dt_diagmerge/dt_tot);
  printf(" Compute Dists = %.4f (%.f %%) \n", dt_recdist/1e3, 100*dt_recdist/dt_tot);
  printf(" Merge Horizontally = %.4f (%.f %%) \n", dt_mergehoriz/1e3, 100*dt_mergehoriz/dt_tot);
  printf(" Merge Vertically  = %.4f (%.f %%) \n", dt_mergever/1e3, 100*dt_mergever/dt_tot);
  printf(" Total = %.4f \n", dt_tot/1e3);
  printf("--------------- Memory usage ----------------\n");
  printf(" Storing norms = %.4f GB \n", (m1-m2)/1e9);
  printf(" Precomputing the sort indices = %.4f GB \n", (free-m1)/1e9);
  printf(" Temporary storage = %.4f GB \n", (m2-m3)/1e9);
  printf("----------------------------- End of leaf-knn -----------------------------\n\n");
  }
}











