
#include <stdio.h> 
#include <stdlib.h>
//#include <cusparse.h>
#include <cuda.h>
#include <helper_cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cuda_profiler_api.h>
#include "knn_seq.cpp"
#define shared_size  8192
#define MaxProcperBlock  1024 





__global__ void compute_norm(int* R, int* C, float* V, int* G_Id, float* K, int M_I, int leaf_batch_g) {

  int row = threadIdx.x;
  int leaf_id_g = leaf_batch_g * gridDim.x + blockIdx.x;

  int g_rowId = leaf_id_g * M_I + row;

  int g_Id = G_Id[g_rowId];
  int ind0_i = R[g_Id];

  int nnz = R[g_Id + 1] - ind0_i;
  float norm_i = 0.0;

  for (int n_i = 0; n_i < nnz; n_i++) norm_i += V[ind0_i + n_i] * V[ind0_i + n_i];
  int ind_write = blockIdx.x * M_I + row;
  K[ind_write] = norm_i;

}

__global__ void compute_dist(int* R, int* C, float* V, int* G_Id, float* Norms, float* K, int* K_Id, int m, int k_nn, int ppl, int leaf_batch_g, int max_nnz, int M){



    // Preproc for Triangular part. 

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
    int col_Id = ind + row_Id - m * (m+1)/2 + (m - row_Id) * ((m - row_Id) + 1)/2;

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


    //int len_ind = b_i * b_j; 

       
    int leaf_id_g = leaf_batch_g * gridDim.z + blockIdx.z;
    
    int g_rowId_I = leaf_id_g * ppl + blockId_I * m + row_Id;
    //int g_rowId_J = leaf_id_g * ppl + blockId_J * m + col_Id;
    
    if (g_rowId_I >= M) return;
    
    int g_Id_i = G_Id[g_rowId_I];
   

    int ind0_i = R[g_Id_i];
    int ind1_i = R[g_Id_i + 1];

 
    int nnz_i = ind1_i - ind0_i;


   
    //float norm_i = Norms[ind_read_norm_i];
    
    __shared__ int si[8192];
    //__shared__ float dist[1024];
    //__shared__ int dist_Id[1024];
    __shared__ float c_tmp[1024];
    float norm_ij;
    if (j == 0) c_tmp[i] = ind0_i;
    __syncthreads();
    int tmp3 = c_tmp[i];
    for (int n_i = j; n_i < nnz_i; n_i += blockDim.x) si[max_nnz * row_Id + n_i] = C[tmp3 + n_i];
    
    __syncthreads();
    
    int k; 
    //dist_Id[j_Id] = G_Id[leaf_id_g * ppl + blockId_j * blockDim.x + j_Id];
     
    int ind_tmp, row_tmp, col_tmp, read_j_Id, read_i_Id, ind_read_i, ind_read_j, nnz_j, ind_norm_i, ind_norm_j; 
    int tmp_0, tmp_1, ind_jk, ret, testInd;

    // for loop over the row points
    //for (int i_tmp = i; i_tmp < m/2; i_tmp += b_i){
    int i_tmp = i;
 
      // for loop over the column points
      for (int j_tmp = 0; j_tmp < m+1; j_tmp++){
        ind_tmp = i_tmp * (m +1) + j_tmp;
        tmp = -8*ind_tmp + 4 * m * (m + 1) - 7;
        row_tmp = sqrt(tmp)/2.0 - 0.5;
        row_tmp = m - 1 - row_tmp;
        col_tmp = ind_tmp + row_tmp - m * (m+1)/2 + (m - row_tmp) * ((m - row_tmp) + 1)/2;


        int shift_i = max_nnz * row_tmp;
        
        if (lower_block) {
          tmp1 = row_tmp;
          row_tmp = col_tmp;
          col_tmp = tmp1;
        }

        c_tmp[ind_tmp] = 0.0;
        ind_norm_i = blockIdx.z * ppl + blockId_I * m + row_tmp;
        ind_norm_j = blockIdx.z * ppl + blockId_J * m + col_tmp;

        //printf("i, j = (%d, %d) -> row,col = (%d, %d) \n", i_tmp, j_tmp, row_tmp, col_tmp);
        //printf("read j %d , read i %d \n", leaf_id_g * ppl + blockId_J * m + col_tmp, leaf_id_g * ppl + blockId_I * m + row_tmp);
        //printf("row,col = (%d, %d) \n", row_tmp, col_tmp);

        read_j_Id = G_Id[leaf_id_g * ppl + blockId_J * m + col_tmp];
        read_i_Id = G_Id[leaf_id_g * ppl + blockId_I * m + row_tmp];
        norm_ij = Norms[ind_norm_i] + Norms[ind_norm_j]; 
        
        ind_read_j = R[read_j_Id];
        ind_read_i = R[read_i_Id];
        nnz_j = R[read_j_Id + 1] - ind_read_j;
        nnz_i = R[read_i_Id + 1] - ind_read_i;
        // loop over the elements 
        
        
        for (int pos = j; pos < nnz_j; pos++){
       		if (j < nnz_i){
            k = C[ind_read_j + pos];
            ret = 0;
            testInd = 0;    
            // binary search 
            for (int l = nnz_i; l > 1; l/=2){
 							tmp_0 = ret + l;
              tmp_1 = nnz_i - 1;
              testInd = (tmp_0 < tmp_1) ? tmp_0 : tmp_1;
              ret = (si[testInd + shift_i] == k) ? testInd : ret; 
            }
             
            tmp_0 = ret + 1;
            tmp_1 = nnz_i - 1;
            testInd = (tmp_0 < tmp_1) ? tmp_0: tmp_1;
            ret = (si[testInd + shift_i] <= k) ? testInd : ret;
            ind_jk = (si[ret + shift_i] == k) ? ret : -1;
            c_tmp[ind] += (ind_jk != -1) ? V[ind_read_j + pos] * V[ind_read_i + j] : 0.0;

        }
        }
         
        __syncthreads();
        
        // reduction 
        
        for (int size = nnz_i/2; size > 0; size /= 2) {
					if (j < size) c_tmp[ind] += c_tmp[ind + size];
          __syncthreads();
        }
        
        int col_write = blockId_J * m + col_tmp;
        int row_write = blockId_I * m + row_tmp;
        int ind_write = blockIdx.z * ppl * ppl + row_write * ppl + col_write;
        int ind_write_T = blockIdx.z * ppl * ppl + col_write * ppl + row_write;
        float val = sqrt(-2*c_tmp[0] + norm_ij);
        if (lower_block == 0 && i_tmp == i && j_tmp == j) K[ind_write] = val;
        if (lower_block == 0 && i_tmp == i && j_tmp == j) K[ind_write_T] = val;
        if (lower_block == 1 && row_tmp != col_tmp && i_tmp == i && j_tmp == j) K[ind_write] = val;
        if (lower_block == 1 && row_tmp != col_tmp && i_tmp == i && j_tmp == j) K[ind_write_T] = val;
        
        //printf("(%d, %d, %d) \n", leaf_id_g, i_tmp, j_tmp);
}

   
}

__global__ void find_neighbor(float* knn, int* knn_Id, float* K, int* K_Id, int* G_Id, int k, int ppl, int leaf_batch_g, int M){

    int col_Id = threadIdx.x; 
    int row_Id = blockIdx.x;

    if (row_Id >= M || col_Id >= M) return;
 
    __shared__ float Dist[4096];
    __shared__ int Dist_Id[4096];

    int size = blockDim.x;
    int leaf_id_g = leaf_batch_g * gridDim.y + blockIdx.y;
    
    int ind_K = blockIdx.z * ppl * (ppl)  + row_Id * (ppl) + col_Id; 
    int i = col_Id;
    //if (row_Id == 1000) printf("val = %.4f , ind = %d \n", K[ind_K], K_Id[ind_K]); 
    Dist[col_Id] = K[ind_K];
    Dist_Id[col_Id] = K_Id[ind_K];
    
    int ind_shared = col_Id;
     
    __syncthreads();

    // bitonic sort
    float tmp_f;
    int tmp_i;
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

  size = 2*k;

  int ind_knn = leaf_id_g * ppl * k + row_Id * k + col_Id;

  // should change to the given knn 
  //if (col_Id >= k && col_Id < size) Dist[col_Id] = 1e30;
  if (col_Id >= k && col_Id < size) Dist[col_Id] = 1e30;
  //if (col_Id >= k && col_Id < size) Dist_Id[col_Id] = 0;
  if (col_Id >= k && col_Id < size) Dist_Id[col_Id] = 0;

  __syncthreads();
	for (int g = 2; g <= size; g *= 2){
		for (int l = g/2; l>0; l /= 2){
		int ixj = i ^ l;
		int ixj_tmp =  ixj;
		if (ixj > i){
			if ((i & g) == 0){
				if (Dist[col_Id] > Dist[ixj_tmp]){
						 tmp_f = Dist[ixj_tmp];
						 Dist[ixj_tmp] = Dist[col_Id];
						 Dist[col_Id] = tmp_f;
						 tmp_i = Dist_Id[ixj_tmp];
						 Dist_Id[ixj_tmp] = Dist_Id[col_Id];
						 Dist_Id[col_Id] = tmp_i;
							}
			} else {
				if (Dist[col_Id] < Dist[ixj_tmp]){
						 tmp_f = Dist[ixj_tmp];
						 Dist[ixj_tmp] = Dist[col_Id];
						 Dist[col_Id] = tmp_f;
						 tmp_i = Dist_Id[ixj_tmp];
						 Dist_Id[ixj_tmp] = Dist_Id[col_Id];
						 Dist_Id[col_Id] = tmp_i;
							}
			  }
	    }
		
		__syncthreads();
		}
     
    if (col_Id < k){
      knn[ind_knn] = Dist[col_Id];
      knn_Id[ind_knn] = Dist_Id[col_Id];
    }
    
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
          //V[ind] = rand()%100;
          V[ind] = ((float) rand()) / (float) RAND_MAX;
        }    
      std::sort(C+R[i], C+(R[i+1]));
      /*
      printf("\n point %d\n", i);
      for (int j=R[i]; j<R[i+1]; j++) {
      printf("(%d ,%.4f) ",C[j], V[j]);
      }
      */
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
  //std::random_shuffle(&G_Id[0], &G_Id[M]);
  /* 
  for (int m = 0; m < M; m++){ 
  printf("G_Id[%d] = %d \n", m , G_Id[m]);
  } 
  */
}

void gpu_knn(int *R, int *C, float *V, int *G_Id, int M, int leaves, int k, float *knn, int *knn_Id, int max_nnz){
 
	int ppl = M/leaves;
	
  size_t free, total;
  cudaMemGetInfo(&free, &total);
  
  int log_size = log2(free / (sizeof(float)));
  double arr_len = pow(2, log_size);  

  int size_batch_leaves = arr_len / (ppl * ppl );
  
  if (size_batch_leaves > leaves) size_batch_leaves = leaves; 
  
  int num_batch_leaves = (leaves) / size_batch_leaves; 
   

	//int M_I = M/leaves;

  float del_t1;
  cudaEvent_t t0; 
  cudaEvent_t t1;
   
  //int m = min(ppl, 1024);
  int m = 32;
  m = (max_nnz > 32) ? min(m, max_nnz) : min(m, 32);
  int num_blocks = (ppl + m - 1) / m;
  
  int block_size_i = m / 2;
  int block_size_j = m + 1;

  int num_batch_I = (ppl + m - 1) / m;
  int num_batch_J = (ppl + m - 1) / m;

  dim3 dimBlock(block_size_j, block_size_i, 1);	
  dim3 dimGrid(num_batch_J, num_batch_I, size_batch_leaves);
 
  dim3 dimBlock_findk(ppl, 1);
  dim3 dimGrid_findk(ppl, size_batch_leaves);

  dim3 dimBlock_norm(ppl);
  dim3 dimGrid_norm(size_batch_leaves);

  cudaMemGetInfo(&free, &total);
  printf("%d kB from  %d kB is free \n", free/1024, total/1024);
  

  float *d_K, *d_Norms;
  int *d_K_Id;
  checkCudaErrors(cudaMalloc((void **) &d_K, sizeof(float) * arr_len));
  //checkCudaErrors(cudaMalloc((void **) &d_K_Id, sizeof(int) * arr_len));
  checkCudaErrors(cudaMalloc((void **) &d_Norms, sizeof(float) * size_batch_leaves * ppl));

  
  //size_t free, total; 

  cudaMemGetInfo(&free, &total);
  printf("%d kB from  %d kB is free \n", free/1024, total/1024);



  checkCudaErrors(cudaEventCreate(&t0));
  checkCudaErrors(cudaEventCreate(&t1));

  checkCudaErrors(cudaEventRecord(t0, 0));
  checkCudaErrors(cudaProfilerStart());
  printf("# leaves : %d \n", leaves);
  printf("# points/leaf : %d \n", ppl);
  printf(" max_nnz : %d \n", max_nnz);
  printf(" blockDim (norms) : (%d) \n", ppl);
  printf(" blockGrid (norms) : (%d) \n", size_batch_leaves); 
  printf(" blockDim (distance) : (%d,%d,1) \n", block_size_j, block_size_i, 1);
  printf(" blockGrid (distance) : (%d,%d,%d) \n", num_batch_J, num_batch_I, size_batch_leaves);
  printf(" blockDim (find knn) : (%d,%d,1) \n", ppl, 1);
  printf(" blockGrid (find knn) : (%d,%d,1) \n", ppl, size_batch_leaves);
  printf(" num leaves per loop : %d \n",size_batch_leaves);
  printf(" # points = %d \n" , M);  
  

  for (int leaf_id_g = 0; leaf_id_g < num_batch_leaves; leaf_id_g++){
    compute_norm <<< dimGrid_norm, dimBlock_norm >>>(R, C, V, G_Id, d_Norms,ppl, leaf_id_g);
    checkCudaErrors(cudaDeviceSynchronize()); 
    compute_dist <<< dimGrid, dimBlock >>>(R, C, V, G_Id, d_Norms, d_K, d_K_Id, m, k, ppl, leaf_id_g, max_nnz, M);
    checkCudaErrors(cudaDeviceSynchronize());
    //find_neighbor <<< dimGrid_findk, dimBlock_findk >>>(knn, knn_Id, d_K, d_K_Id, G_Id, k, ppl, leaf_id_g, M);
  } 

  checkCudaErrors(cudaProfilerStop());  
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaEventRecord(t1, 0));
  checkCudaErrors(cudaEventSynchronize(t1));
  checkCudaErrors(cudaEventElapsedTime(&del_t1, t0, t1));
  


   
  printf("\n Elapsed time (s) : %.4f \n ", del_t1/1000);
 
  checkCudaErrors(cudaFree(d_K));
  checkCudaErrors(cudaEventDestroy(t0));
  checkCudaErrors(cudaEventDestroy(t1));

}





int main(int argc, char **argv)
{

  //, del_t2, del_t3;

    checkCudaErrors(cudaSetDevice(0));

    int d, nnzperrow;
    float *h_V, *d_V;
    int *h_C, *d_C;
    int *h_R, *d_R;
    int *h_G_Id, *d_G_Id;
    int M = 1024;     // total number of points 
    int leaves = 1;     // number of leaves
    d = 10000;
    int k = 32;
    nnzperrow = 32;
    int max_nnz = nnzperrow;
    int leaf_size = M / leaves; 
    

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
    /* 
    for (int i = 0; i < M; i++){
        int nnz = h_R[i+1] - h_R[i];
        for (int j = 0; j < nnz; j++)
        printf("R[%d] = %d , C[%d] = %d , V[%d] = %.4f \n", i ,h_R[i], h_R[i]+j, h_C[h_R[i] + j], h_R[i]+j, h_V[h_R[i]+j]);
    }    
    */
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

    printf("\nRandom csr is generated  \n");

    gpu_knn(d_R, d_C, d_V, d_G_Id, M, leaves, k, d_knn, d_knn_Id, max_nnz);
  
    checkCudaErrors(cudaMemcpy(h_knn, d_knn, sizeof(float) * M * k, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_knn_Id, d_knn_Id, sizeof(int) * M * k, cudaMemcpyDeviceToHost));
 
    int test_leaf = 0;
    int test_pt = 1000;
    //printf(" \n running Seq knn \n");
    //printf("\n test for leaf %d \n",test_leaf);

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
      match = (h_knn_Id_seq[test_pt*k + i] == h_knn_Id[ind]);
      //printf("seq val %d,\t gpu_val %d , \t match %d , \t v_seq %.4f, \t v_gpu %.4f , \t ind = %d\n", h_knn_Id_seq[test_pt*k + i], h_knn_Id[ind], match, h_knn_seq[test_pt*k + i], h_knn[ind], ind);
      if (match) acc += 1.0;
      if (counter < 2 && match==0) {
        counter++;
        gpu_pt = h_knn_Id[ind];
        seq_pt = h_knn_Id_seq[test_pt * k + i];
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

    printf("accuracy %.2f %% for leaf %d\n\n", acc*100, test_leaf);







 
    printf("\n\n");
    checkCudaErrors(cudaFree(d_R));
    checkCudaErrors(cudaFree(d_G_Id));
    checkCudaErrors(cudaFree(d_C));
    checkCudaErrors(cudaFree(d_V));
    free(h_R);
    free(h_C);
    free(h_V);
    free(h_G_Id);


}







