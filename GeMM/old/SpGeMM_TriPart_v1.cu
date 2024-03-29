
#include <stdio.h> 
#include <stdlib.h>
//#include <cusparse.h>
#include <cuda.h>
#include <helper_cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include "knn_seq.cpp"



__global__ void compute_norm(int* R, int* C, float* V, int* G_Id, float* Norms, int ppl) {

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
  //printf("g_rowId = %d , val = %.4f \n", g_rowId, norm_i); 
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


    //int ind_read_norm_I = blockIdx.z * ppl + blockId_I * m + row_Id;
    //int ind_read_norm_J = blockIdx.z * ppl + blockId_J * m + col_Id;
     
    //norm_ij += Norms[ind_read_norm_I] + Norms[ind_read_norm_J]; 
    norm_ij += Norms[g_rowId_I] + Norms[g_rowId_J]; 

    //norm_ij += K[ind_read_norm_J]; 
    int blockdim_comp_x = (lower_block) ? row_Id+1 : m - row_Id; 

    //for (int n_i = start_i; n_i < nnz_i; n_i += blockdim_comp_x) si[shift_i + n_i] = C[ind0_i + n_i];
 
    __syncthreads();

    float c_tmp = 0.0;
    float c;
    int tmp_0, tmp_1, ind_jk, k, ret, testInd;
    
    ret=0; 
    testInd = 0;

    /*
    for (int pos_k=0; pos_k<nnz_j;pos_k++){       
        
        k = C[ind0_j + pos_k];
            
        // Binary search 
        //for (int l=nnz_i-ret; l > 1; l/=2){
        for (int l=nnz_i; l > 1; l/=2){
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
    //if (g_rowId_I == 10 && g_rowId_J == 10 ) printf("norm =%.4f, inner = %.4f \n", norm_ij, 2*c_tmp);
    */
    c = -2*c_tmp + norm_ij;
    c_tmp = (c > 0) ? sqrt(c) : 0.0;
    
    //if (leaf_id_g == 1000 && blockId_J * m + col_Id == 1000) printf(" %.4f , (%d,%d) , n %.4f \n ", c_tmp , row_Id, col_Id, norm_ij);
	  int col_write = blockId_J * m + col_Id; 
	  int row_write = blockId_I * m + row_Id;
    //int ind_write = blockIdx.z * ppl * ppl + row_write * ppl + col_write;


	  //int col_write_T = row; 
	  //int row_write_T = blockId_J * m + col_Id;
	  //int ind_write_T = blockIdx.z * ppl * ppl + col_write * ppl + row_write;
    

    
    //int max_ind = gridDim.z * ppl * ppl;
    //if (ind_write >= max_ind) printf("batchsize = %d, batch =  %d , blockIdx.z = %d, row_write = %d, col_write = %d  , max_ind = %d , ind_write_T \n",gridDim.z, leaf_batch_g, blockIdx.z, row_write, col_write, max_ind, ind_write);
    //if (ind_write_T >= max_ind) printf("batchsize = %d, batch =  %d , blockIdx.z = %d, row_write = %d, col_write = %d  \n",gridDim.z, leaf_batch_g, blockIdx.z, row_write, col_write);
    //if (ind_write >= e) printf("blockDim.z = %d , leaf_batch_g = %d, block = %d , row_write = %d , col_write = %d , \n", gridDim.z, leaf_batch_g, blockIdx.z, row_write, col_write);
    //if (ind_write_T >= e) printf("blockDim.z = %d , leaf_batch_g = %d , leafId T block = %d , row_write = %d , col_write = %d \n", gridDim.z , leaf_batch_g, blockIdx.z, row_write, col_write);
    
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

      //int ind_K = blockIdx.y * ppl * ppl + (row_Id) * ppl + col_Id + batch_ppl_col * len; 
      int i = col_Id;
         
      Dist[col_Id] = K[blockIdx.y * ppl * ppl + (row_Id) * ppl + col_Id + batch_ppl_col * len];
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
    

    //size = (ppl/1024) *k;
    


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

    if (col_Id < k){
      knn[leaf_id_g * ppl * k + row_Id * k + col_Id] = res_Dist[col_Id];
      knn_Id[leaf_id_g * ppl * k + row_Id * k + col_Id] = res_Dist_Id[col_Id];
    }

}



void gen_sparse(int M, int tot_nnz, int d, int *R, int *C, float *V) {
 
    int nnz_row;
    int val;

    for (int i=0; i < M; i++){
      nnz_row = R[i+1] - R[i];
      for (int j=0; j < nnz_row; j++){
          int ind = R[i]+j; 
          val = rand()%d;
          //val = rand()%d;
          C[ind] = val;
          //V[ind] = (rand()%1000)/30;
          V[ind] = ((float) rand()) / (float) RAND_MAX;
        }    
      std::sort(C+R[i], C+(R[i+1]));
    }
}

void gen_R(int M, int nnzperrow, int *R, int *G_Id, int d) {  
  R[0] = 0;
  int tot_nnz = 0;
  int val;
  for (int m =1; m <= M; m++){ 
   //val = 1 + rand()%(2*nnzperrow);
   val = nnzperrow; //+ rand()%nnzperrow;
   if (val > d) val = 1; 
   tot_nnz += val;
   R[m] = tot_nnz;
   G_Id[m-1] = m-1;
  } 
  std::random_shuffle(&G_Id[0], &G_Id[M]);
  /*
  for (int m = 0; m < M; m++){ 
  printf("G_Id[%d] = %d \n", m , G_Id[m]);
  } 
  */
}

void gpu_knn(int *R, int *C, float *V, int *G_Id, int M, int leaves, int k, float *knn, int *knn_Id, int max_nnz){
 
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
  compute_norm <<< dimGrid_norm, dimBlock_norm >>>(R, C, V, G_Id, d_Norms, ppl);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaEventRecord(t0, 0));
  for (int leaf_id_g = 0; leaf_id_g < num_batch_leaves; leaf_id_g++){
    printf("running batchleaf=%d \n", leaf_id_g);
    
    
    compute_dist <<< dimGrid, dimBlock >>>(R, C, V, G_Id, d_Norms, d_K, m , k, ppl, leaf_id_g, max_nnz, M);
    checkCudaErrors(cudaDeviceSynchronize());
    //find_neighbor <<< dimGrid_findknn, dimBlock_findknn >>>(knn, knn_Id, d_K, G_Id, k, ppl, m, leaf_id_g, M);
    //checkCudaErrors(cudaDeviceSynchronize());
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
	printf("blockDim (distance) : (%d,%d,1) \n", block_size_j, block_size_i);
  printf("blockGrid (distance) : (%d,%d,%d) \n", num_batch_J, num_batch_I, size_batch_leaves);
	printf("blockDim (find knn) : (%d, 1) \n", ppl);
  printf("blockGrid (find knn) : (%d,%d) \n", ppl, num_batch_leaves);
  printf("\n Elapsed time (s) : %.4f \n ", del_t1/1000);
  printf("\n Memory for norms %.4f GB \n", (m1-free)/1e9);
  printf("\n Memory for temp storage %.4f GB \n", (free-m2)/1e9);

  printf(" # points = %d" , M);
 
  checkCudaErrors(cudaFree(d_K));
  checkCudaErrors(cudaEventDestroy(t0));
  checkCudaErrors(cudaEventDestroy(t1));

}





int main(int argc, char **argv)
{

    checkCudaErrors(cudaSetDevice(0));

    int d, nnzperrow;
    float *h_V, *d_V;
    int *h_C, *d_C;
    int *h_R, *d_R;
    int *h_G_Id, *d_G_Id;
    int M = 2048 * 2048;
    int leaves = 2048;
    d = 100000;
    int k = 32;
    nnzperrow = 32;
    int max_nnz = nnzperrow;
    int leaf_size = M / leaves; 
    

    bool print_pt = false;    
    bool print_res = true;    
    int test_leaf = 1000;    
    int test_pt = 2000;

    int *d_knn_Id, *h_knn_Id, *h_knn_Id_seq;
    float *d_knn, *h_knn, *h_knn_seq;

    h_R = (int *)malloc(sizeof(int)*(M+1));
    h_G_Id = (int *)malloc(sizeof(int)*(M));

    h_knn = (float *)malloc(sizeof(float) * M *k);
    h_knn_seq = (float *)malloc(sizeof(float) * M *k / leaves);
    h_knn_Id = (int *)malloc(sizeof(int) * M *k);
    h_knn_Id_seq = (int *)malloc(sizeof(int) * M *k / leaves);

    // generate random data 
    gen_R(M, nnzperrow, h_R,h_G_Id, d);
    int tot_nnz = h_R[M];
		h_V = (float *)malloc(sizeof(float)*tot_nnz);
    h_C = (int *)malloc(sizeof(int)*tot_nnz);
    gen_sparse(M, tot_nnz, d , h_R, h_C, h_V);   
    if (print_pt){   
    for (int i = 0; i < M; i++){
        int nnz = h_R[i+1] - h_R[i];
        for (int j = 0; j < nnz; j++)
        printf("R[%d] = %d , C[%d] = %d , V[%d] = %.4f \n", i ,h_R[i], h_R[i]+j, h_C[h_R[i] + j], h_R[i]+j, h_V[h_R[i]+j]);
    }    
    }
    checkCudaErrors(cudaMalloc((void **) &d_R, sizeof(int)*(M+1)));
    checkCudaErrors(cudaMalloc((void **) &d_G_Id, sizeof(int)*(M)));
    checkCudaErrors(cudaMalloc((void **) &d_C, sizeof(int)*tot_nnz));
    checkCudaErrors(cudaMalloc((void **) &d_V, sizeof(float)*tot_nnz));
    checkCudaErrors(cudaMalloc((void **) &d_knn_Id, sizeof(int)*M*k));
    checkCudaErrors(cudaMalloc((void **) &d_knn, sizeof(float)*M*k));
 
    checkCudaErrors(cudaMemcpy(d_C, h_C, sizeof(int)*tot_nnz, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_V, h_V, sizeof(float)*tot_nnz, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_R, h_R, sizeof(int)*(M+1), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_G_Id, h_G_Id, sizeof(int)*(M), cudaMemcpyHostToDevice));

    printf("Random csr is generated  \n");
    cudaEvent_t t0; 
    cudaEvent_t t1;
    float del_t1;

    checkCudaErrors(cudaEventCreate(&t0));
    checkCudaErrors(cudaEventCreate(&t1));
    checkCudaErrors(cudaEventRecord(t0, 0));


    gpu_knn(d_R, d_C, d_V, d_G_Id, M, leaves, k, d_knn, d_knn_Id, max_nnz);
    
    
    checkCudaErrors(cudaEventRecord(t1, 0));
    checkCudaErrors(cudaEventSynchronize(t1));
    checkCudaErrors(cudaEventElapsedTime(&del_t1, t0, t1));

    printf("\n Elapsed time (s) : %.4f \n ", del_t1/1000);


    checkCudaErrors(cudaMemcpy(h_knn, d_knn, sizeof(float) * M * k, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_knn_Id, d_knn_Id, sizeof(int) * M * k, cudaMemcpyDeviceToHost));
  

    printf(" \n running Seq knn \n");
    printf("\n test for leaf %d , pt %d\n",test_leaf, test_pt);  

    f_knnSeq(h_R, h_C, h_V, h_G_Id, h_knn_seq, h_knn_Id_seq, k, test_leaf, test_pt, leaf_size);
    
    float acc= 0.0;  

    
    int ind;
    bool match;
    int counter = 0;
    int gpu_pt,seq_pt,ind_seq,ind_gpu, nnz_gpu,nnz_seq;
    int ind0_i = h_R[h_G_Id[test_leaf * leaf_size + test_pt]];
    int nnz_i = h_R[h_G_Id[test_leaf * leaf_size + test_pt] + 1] - ind0_i;
    //for (int i=0; i < nnz_i; i++) printf("[(%d, %d, %.4f)] \n", h_G_Id[test_leaf * leaf_size + test_pt], h_C[ind0_i + i], h_V[ind0_i + i]);

    for (int i = 0; i < k; i++){
      ind = test_leaf * k * leaf_size + test_pt * k + i;
      //match = (h_knn_Id_seq[test_pt*k + i] == h_knn_Id[ind]);
      match = (h_knn_Id_seq[i] == h_knn_Id[ind]);
      if (print_res){
      //printf("seq ind %d,\t gpu_ind %d , \t match %d , \t v_seq %.4f, \t v_gpu %.4f , \t ind = %d\n", h_knn_Id_seq[test_pt*k + i], h_knn_Id[ind], match, h_knn_seq[test_pt*k + i], h_knn[ind], ind);
      printf("seq ind %d,\t gpu_ind %d , \t match %d , \t v_seq %.4f, \t v_gpu %.4f , \t ind = %d\n", h_knn_Id_seq[i], h_knn_Id[ind], match, h_knn_seq[i], h_knn[ind], ind);
      }
      if (match) acc += 1.0;
      if (counter < 2 && match==0) {
        counter++;
		    gpu_pt = h_knn_Id[ind];
        //seq_pt = h_knn_Id_seq[test_pt * k + i];
        seq_pt = h_knn_Id_seq[i];
        ind_gpu = h_R[gpu_pt];
        ind_seq = h_R[seq_pt];
        nnz_gpu = h_R[gpu_pt + 1]  - h_R[gpu_pt];
        nnz_seq = h_R[seq_pt + 1]  - h_R[seq_pt]; 
        //printf("gpu pt %d \n", gpu_pt); 
       
        //for (int q=0; q < nnz_gpu; q++) printf("[(%d, %d, %.4f)] \n", gpu_pt, h_C[ind_gpu + q], h_V[ind_gpu + q]);
        //printf("\n seq pt %d \n", seq_pt); 
       
        //for (int q=0; q < nnz_seq; q++) printf("[(%d, %d, %.4f)] \n", seq_pt, h_C[ind_seq + q], h_V[ind_seq + q]);
        
		
		}
    }
    
    acc /= k;    
    printf("\n\naccuracy %.4f for leaf %d\n\n", acc*100, test_leaf);
    
    checkCudaErrors(cudaFree(d_R));
    checkCudaErrors(cudaFree(d_G_Id));
    checkCudaErrors(cudaFree(d_C));
    checkCudaErrors(cudaFree(d_V));
 
    free(h_R);
    free(h_C);
    free(h_V);
    free(h_G_Id);
    free(h_knn);
    free(h_knn_Id);
    free(h_knn_seq);
    free(h_knn_Id_seq);


}
