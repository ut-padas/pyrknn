
#include "TSKNN_sparse.h"


__global__ void TSKNN_compute_norm(int* R, int* C, float* V, int* G_Id, float* Norms, int ppl) {

  int row = threadIdx.x + blockIdx.x * blockDim.x;
  int leaf_id_g = blockIdx.y;
  
  int g_rowId = leaf_id_g * ppl + row;

  int g_Id = G_Id[g_rowId]; 
  int ind0_i = R[g_Id];
 
  int nnz = R[g_Id + 1] - ind0_i;
  float norm_i = 0.0;
  
  for (int n_i = 0; n_i < nnz; n_i++) norm_i += V[ind0_i + n_i] * V[ind0_i + n_i];
  int ind_write = blockIdx.y * ppl + row;
  Norms[ind_write] = norm_i;
}

__global__ void compute_dist(int* R, int* C, float* V, int* G_Id, float* Norms, float* K, int m , int k_nn, int ppl, int leaf_batch_g, int max_nnz, int M){


    // preproc for triangular partitioning 

    int b_i = blockIdx.y;
    int b_j = blockIdx.x;
  
    int b_ind = b_i * gridDim.x + b_j;  
    int num_blocks = ppl / m; 
  
    float tmp = num_blocks * num_blocks - b_ind - 1;
    int blockId_I = sqrt(tmp);
    blockId_I = num_blocks - 1 - blockId_I;

    int blockId_J_tmp = b_ind - num_blocks*num_blocks + (num_blocks - blockId_I)*(num_blocks - blockId_I) + 2*blockId_I;
    int blockId_J = (blockId_J_tmp+1)/2;
 
    int i = threadIdx.y;
    int j = threadIdx.x; 
    
    int ind = i * (m + 1) + j; 
     
    tmp = -8*ind + 4 * m * (m + 1) - 7;
    int row_Id = sqrt(tmp)/2.0 - 0.5;
    row_Id = m - 1 - row_Id;
    int col_Id = ind + row_Id - m *(m +1)/2 + (m - row_Id) * ((m - row_Id) + 1)/2; 
     
    int tmp1;
    bool lower_block = false;

    // determine the lower block
    if (blockId_I % 2 == 0 && blockId_J_tmp % 2 != 0){
      tmp1 = row_Id;
      row_Id = col_Id; 
      col_Id = tmp1;
      lower_block = true;
    }
    if (blockId_I % 2 != 0 && blockId_J_tmp % 2 != 0){
      tmp1 = row_Id; 
      row_Id = col_Id;
      col_Id = tmp1;
      lower_block = true;
    }
     
    // end of partioning    
    int leaf_id_g = leaf_batch_g * gridDim.z + blockIdx.z;
     
    int g_rowId_I = leaf_id_g * ppl + blockId_I * m + row_Id;
    int g_rowId_J = leaf_id_g * ppl + blockId_J * m + col_Id;
    int leaves = M / ppl;
    if (g_rowId_I >= M || g_rowId_J >= M) return;
    if (leaf_id_g >= leaves) return;
    
    int g_Id_i = G_Id[g_rowId_I]; 
    int g_Id_j = G_Id[g_rowId_J];     
    
    int ind0_i = R[g_Id_i];
    int ind1_i = R[g_Id_i + 1];

    int ind0_j = R[g_Id_j];
    int ind1_j = R[g_Id_j + 1];

    
 
    int nnz_i = ind1_i - ind0_i;
    int nnz_j = ind1_j - ind0_j;


    float norm_ij = 0.0;    

    __shared__ int si[8192];

    
    int shift_i = max_nnz * row_Id;

    int start_i; 

    start_i = (lower_block) ? col_Id : col_Id-row_Id;

     
    //norm_ij += Norms[ind_read_norm_I] + Norms[ind_read_norm_J]; 
    norm_ij += Norms[g_rowId_I] + Norms[g_rowId_J]; 

    int blockdim_comp_x = (lower_block) ? row_Id+1 : m - row_Id; 

    for (int n_i = start_i; n_i < nnz_i; n_i += blockdim_comp_x) si[shift_i + n_i] = C[ind0_i + n_i];
 
    __syncthreads();

    float c_tmp = 0.0;
    //float c;
    int tmp_0, tmp_1, ind_jk, k, ret, testInd;
    
    ret=0; 
    testInd = 0;

    
    for (int pos_k=0; pos_k<nnz_j;pos_k++){       
        
        k = C[ind0_j + pos_k];
            
        // Binary search 
        for (int l=nnz_i-ret; l > 1; l/=2){
            tmp_0 = ret+l;
            tmp_1 = nnz_i-1;
            testInd = (tmp_0 < tmp_1) ? tmp_0: tmp_1;
            ret = (si[testInd+ shift_i] <= k) ? testInd : ret ;
            //ret = (C[testInd + ind0_i] <= k) ? testInd : ret ;
        }
        tmp_0 = ret+1;
        tmp_1 = nnz_i-1;
        testInd = (tmp_0 < tmp_1) ? tmp_0: tmp_1;
        ret = (si[testInd + shift_i] <= k) ? testInd : ret;
        //ret = (C[testInd + ind0_i] <= k) ? testInd : ret;
        ind_jk = (si[ret + shift_i] == k) ? ret : -1;
        //ind_jk = (C[ret + ind0_i] == k) ? ret : -1;
        c_tmp += (ind_jk != -1) ? V[ind0_j + pos_k]*V[ind0_i + ind_jk] : 0;
        
    }
    if (leaf_id_g == 0 && blockId_I * m + row_Id == 3 && blockId_J * m + col_Id == 3) printf("pos = %d , (%.4f, %.4f) \n", col_Id, norm_ij, c_tmp);
    c_tmp = -2*c_tmp + norm_ij;
    c_tmp = (c_tmp > 0) ? sqrt(c_tmp) : 0.0;
    
	  int col_write = blockId_J * m + col_Id; 
	  int row_write = blockId_I * m + row_Id;

    //__syncthreads(); 
    if (lower_block == 0) K[blockIdx.z * ppl * ppl + row_write * ppl + col_write] = c_tmp; 
    if (lower_block == 0) K[blockIdx.z * ppl * ppl + col_write * ppl + row_write] = c_tmp; 
    if (lower_block == 1 && row_Id != col_Id) K[blockIdx.z * ppl * ppl + row_write * ppl + col_write] = c_tmp;
    if (lower_block == 1 && row_Id != col_Id) K[blockIdx.z * ppl * ppl + col_write * ppl + row_write] = c_tmp;
    
}

__global__ void find_neighbor(float* knn, int* knn_Id, float* K, int* G_Id, int k, int ppl, int m, int leaf_batch_g, int M){

    int col_Id = threadIdx.x; 
    int row_Id = blockIdx.x;

    if (row_Id >= M || col_Id >= M) return;
 
    __shared__ float Dist[1024];
    __shared__ int Dist_Id[1024];
    __shared__ float res_Dist[4096];
    __shared__ int res_Dist_Id[4096];


    int size = blockDim.x;
    int leaf_id_g = leaf_batch_g * gridDim.y + blockIdx.y; 
    int len = blockDim.x;
    float tmp_f;
    int tmp_i;
    for (int batch_ppl_col = 0; batch_ppl_col < ppl / len; batch_ppl_col++){ 

      int i = col_Id;
         
      Dist[col_Id] = K[blockIdx.y * ppl * ppl + row_Id * ppl + col_Id + batch_ppl_col * len];
      Dist_Id[col_Id] = G_Id[leaf_id_g * ppl + col_Id + batch_ppl_col * len];
      int ind_shared = col_Id;
      __syncthreads();
    
      // bitonic sort
      for (int g = 2; g <= size; g *= 2){
        for (int l = g/2; l>0; l /= 2){
          int ixj = i ^ l;
          int ixj_tmp = ixj;
          if (ixj > i){
            if ((i & g) == 0){
              if (Dist[ind_shared] > Dist[ixj_tmp]){
                tmp_f = Dist[ixj_tmp];
                Dist[ixj_tmp] = Dist[ind_shared];
                Dist[ind_shared] = tmp_f;
                tmp_i = Dist_Id[ixj_tmp];
                Dist_Id[ixj_tmp] = Dist_Id[ind_shared];
                Dist_Id[ind_shared] = tmp_i;
              }
            } else {
              if (Dist[ind_shared] < Dist[ixj_tmp]){
                tmp_f = Dist[ixj_tmp];
                Dist[ixj_tmp] = Dist[ind_shared];
                Dist[ind_shared] = tmp_f;
                tmp_i = Dist_Id[ixj_tmp];
                Dist_Id[ixj_tmp] = Dist_Id[ind_shared];
                Dist_Id[ind_shared] = tmp_i;
              }
            }
          }
       
          __syncthreads();
        }
      }
    
      if (col_Id <k) res_Dist[batch_ppl_col * k + col_Id] = Dist[col_Id];
      if (col_Id <k) res_Dist_Id[batch_ppl_col * k + col_Id] = Dist_Id[col_Id];
      __syncthreads();
    }
    

    //size = (ppl/len) *k;
    


    size = k + (ppl/len) *k;
 
    //int ind_knn = leaf_id_g * ppl * k + row_Id * k + col_Id;

    if (col_Id >= k && col_Id < size) res_Dist[col_Id + (ppl/len) *k] = 1.0e30;
    if (col_Id >= k && col_Id < size) res_Dist_Id[col_Id + (ppl/len) * k] = 0;
    
    // should be replaced for the correct knn
    //if (col_Id >= k && col_Id < size) Dist[col_Id] = knn[ind_knn];
    //if (col_Id >= k && col_Id < size) Dist_Id[col_Id] = knn_Id[ind_knn];

    __syncthreads();

    int i = col_Id;
	
    for (int g = 2; g <= size; g *= 2){
		  for (int l = g/2; l>0; l /= 2){
	  	  int ixj = i ^ l;
  		  if (ixj > i){
	  		  if ((i & g) == 0){
		  		  if (res_Dist[col_Id] > res_Dist[ixj]){

						  tmp_f = res_Dist[ixj];
	 					  res_Dist[ixj] = res_Dist[col_Id];
    				  res_Dist[col_Id] = tmp_f;
 
  					  tmp_i = res_Dist_Id[ixj];
	 	 				  res_Dist_Id[ixj] = res_Dist_Id[col_Id];
						  res_Dist_Id[col_Id] = tmp_i;
					  }
		  	  } else {
			  	  if (res_Dist[col_Id] < res_Dist[ixj]){

						  tmp_f = res_Dist[ixj];
						  res_Dist[ixj] = res_Dist[col_Id];
						  res_Dist[col_Id] = tmp_f;

			  			tmp_i = res_Dist_Id[ixj];
						  res_Dist_Id[ixj] = res_Dist_Id[col_Id];
						  res_Dist_Id[col_Id] = tmp_i;
						}
			    }
			  }
		    __syncthreads();
      } 
    }

		//__syncthreads();
    if (col_Id < k){
      if (leaf_id_g == 0 && row_Id == 3) printf("pos = %d , (%.4f, %d) \n", col_Id, res_Dist[col_Id], res_Dist_Id[col_Id]);
      int ind = leaf_id_g * ppl * k + row_Id * k + col_Id;
      if (ind == 13) printf("leaf = %d , row_Id = %d , col_Id = %d , (%.4f , %d) \n", leaf_id_g, row_Id, col_Id, res_Dist[col_Id], res_Dist_Id[col_Id]);
      knn[leaf_id_g * ppl * k + row_Id * k + col_Id] = res_Dist[col_Id];
      knn_Id[leaf_id_g * ppl * k + row_Id * k + col_Id] = res_Dist_Id[col_Id];
      if (ind == 13) printf(" print %.4f , %d \n", knn[13], knn_Id[13]);
    }

}



void TSKNN_gpu(int *R, int *C, float *V, int *G_Id, int M, int leaves, int k, float *knn, int *knn_Id, int max_nnz){
 
	int ppl = M/leaves;
	int m = 8192 / max_nnz;
  m = min(m, ppl);

  
  if (m > 32){ 
    m = 32; 
  } 


	int num_batch_I = (ppl + m - 1) / m;
	int num_batch_J = (ppl + m - 1) / m;

  size_t free, total, m1, m2;

  float *d_K, *d_Norms;
  cudaMemGetInfo(&m1, &total);
  checkCudaErrors(cudaMalloc((void **) &d_Norms, sizeof(float) * ppl * leaves));

  cudaMemGetInfo(&free, &total);


  size_t size_leaf = sizeof(float) * ppl * ppl;
  
  int size_batch_leaves = free / size_leaf;
  size_batch_leaves /= 2;
  size_batch_leaves *= 2;
  if (size_batch_leaves > leaves) size_batch_leaves = leaves;
  int num_batch_leaves = (leaves + size_batch_leaves - 1) / size_batch_leaves;


  
  float del_t1;
  cudaEvent_t t0; 
  cudaEvent_t t1;
  int block_size_i = m / 2;
  int block_size_j = m + 1; 
  

  dim3 dimBlock(block_size_j, block_size_i, 1);	
  dim3 dimGrid(num_batch_J, num_batch_I, size_batch_leaves); 
  
  int t_b = (ppl > 1024) ? 1024 : ppl;
   
  dim3 dimBlock_findknn(t_b, 1);
  dim3 dimGrid_findknn(ppl, size_batch_leaves);

  int num_batch_norm = (ppl > 1024) ? ppl/1024 : 1; 

  dim3 dimBlock_norm(t_b, 1);	
  dim3 dimGrid_norm(num_batch_norm, leaves); 
  
  

  
  
  size_t size = size_leaf * size_batch_leaves;
  checkCudaErrors(cudaMalloc((void **) &d_K, size));
  cudaMemGetInfo(&m2, &total);
  
  printf("\n Memory for norms %.4f GB \n", (m1-free)/1e9);
  printf("\n Memory for temp storage %.4f GB \n", (free-m2)/1e9);

  checkCudaErrors(cudaEventCreate(&t0));
  checkCudaErrors(cudaEventCreate(&t1));
  TSKNN_compute_norm <<< dimGrid_norm, dimBlock_norm >>>(R, C, V, G_Id, d_Norms, ppl);
 
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaEventRecord(t0, 0));
 
  for (int leaf_id_g = 0; leaf_id_g < num_batch_leaves; leaf_id_g++){
 
    printf("running batchleaf=%d \n", leaf_id_g);
     
    compute_dist <<< dimGrid, dimBlock >>>(R, C, V, G_Id, d_Norms, d_K, m , k, ppl, leaf_id_g, max_nnz, M);
    checkCudaErrors(cudaDeviceSynchronize());
    find_neighbor <<< dimGrid_findknn, dimBlock_findknn >>>(knn, knn_Id, d_K, G_Id, k, ppl, m, leaf_id_g, M);
    checkCudaErrors(cudaDeviceSynchronize());
  } 
  
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaEventRecord(t1, 0));
  checkCudaErrors(cudaEventSynchronize(t1));
  checkCudaErrors(cudaEventElapsedTime(&del_t1, t0, t1));
 
  checkCudaErrors(cudaEventRecord(t0, 0));
  checkCudaErrors(cudaEventRecord(t1, 0));
  checkCudaErrors(cudaEventSynchronize(t1));
  checkCudaErrors(cudaEventElapsedTime(&del_t1, t0, t1));

  printf("# leaves : %d \n", leaves);
  printf("# points/leaf : %d \n", ppl);
  printf("  max_nnz : %d \n", max_nnz);
  printf("blockDim (norm) : (%d,%d) \n", t_b, 1);
  printf("blockGrid (norm) : (%d,%d) \n", num_batch_norm, leaves);
	printf("blockDim (distance) : (%d,%d,1) \n", block_size_j, block_size_i);
  printf("blockGrid (distance) : (%d,%d,%d) \n", num_batch_J, num_batch_I, size_batch_leaves);
	printf("blockDim (find knn) : (%d, 1) \n", t_b);
  printf("blockGrid (find knn) : (%d,%d) \n", ppl, num_batch_leaves);
  printf("\n Elapsed time (s) : %.4f \n ", del_t1/1000);
  printf("\n Memory for norms %.4f GB \n", (m1-free)/1e9);
  printf("\n Memory for temp storage %.4f GB \n", (free-m2)/1e9);

  printf(" # points = %d" , M);
 
  checkCudaErrors(cudaFree(d_K));
  checkCudaErrors(cudaFree(d_Norms));
  checkCudaErrors(cudaEventDestroy(t0));
  checkCudaErrors(cudaEventDestroy(t1));

}





