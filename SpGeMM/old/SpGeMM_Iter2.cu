
#include <stdio.h> 
#include <stdlib.h>
//#include <cusparse.h>
#include <cuda.h>
#include <helper_cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include "knn_seq.cpp"
#include <limits.h>


__global__ void compute_norm(int* R, int* C, float* V, int* G_Id, float* Norms, int ppl) {

  int row = threadIdx.x;
  int leaf_id_g = blockIdx.x;
  
  int g_rowId = leaf_id_g * ppl + row;

  int g_Id = G_Id[g_rowId]; 
  int ind0_i = R[g_Id];
 
  int nnz = R[g_Id + 1] - ind0_i;
  float norm_i = 0.0;
  
  for (int n_i = 0; n_i < nnz; n_i++) norm_i += V[ind0_i + n_i] * V[ind0_i + n_i];
  int ind_write = blockIdx.x * ppl + row;
  Norms[ind_write] = norm_i;

}

__global__ void knn_kernel_tri(int* R, int* C, float* V, int* G_Id, float* Norms , int k_nn, float* KNN_dist, int* KNN_Id, int ppl, int max_nnz, int m) {

    __shared__ int SM[4096];
    __shared__ float SM_dist[2048];
    
      int ind = threadIdx.x;
      int leaf_id_g = blockIdx.y;  
      int block = blockIdx.x;

      
      int i = ind / m;
      int j = ind - i * m;

      int g_rowId = leaf_id_g * ppl + block * m + i;
      int g_colId = leaf_id_g * ppl + block * m + j;

      int perm_i = G_Id[g_rowId];
      int perm_j = G_Id[g_colId];

      int ind0_i = R[perm_i];
      int ind1_i = R[perm_i + 1];

      int ind0_j = R[perm_j];
      int ind1_j = R[perm_j + 1];
     
      int nnz_i = ind1_i - ind0_i;
      int nnz_j = ind1_j - ind0_j;
     
      
      float norm_ij = 0.0;
      if (nnz_i > 256 || nnz_j > 256) printf("Exceeding the max nnz/pt \n");  

      
      norm_ij += Norms[g_rowId] + Norms[g_colId];

      int shift_i = max_nnz * i;
       
      // TODO: this reading should be balanced among the threads
      // above problem is fixed
      
      for (int n_i = j; n_i < nnz_i; n_i += m) SM[shift_i + n_i] = C[ind0_i + n_i];
      __syncthreads();
      
      float c_tmp = 0.0;
      int tmp_0, tmp_1, ind_jk, k, ret, testInd;
 
      ret = 0;
      testInd = 0;
      if (j >= i){
      // loop over the elements of j
      for (int pos_k = 0; pos_k < nnz_j; pos_k++){
        
        k = SM[max_nnz * j + pos_k];

        // Binary search
        for (int l = nnz_i - ret; l > 1; l /= 2){
          tmp_0 = ret + l;
          tmp_1 = nnz_i - 1;
          testInd = (tmp_0 < tmp_1) ? tmp_0 : tmp_1;
          ret = (SM[testInd + shift_i] <= k) ? testInd : ret;
        }
        
        tmp_0 = ret + 1;
        tmp_1 = nnz_i - 1;
        testInd = (tmp_0 < tmp_1 ) ? tmp_0 : tmp_1;
        ret = (SM[testInd + shift_i] <= k) ? testInd : ret; 
        ind_jk = (SM[ret + shift_i] == k) ? ret : -1;
        c_tmp += (ind_jk != -1) ? V[ind0_j + pos_k] * V[ind0_i + ind_jk] : 0;
      }
      c_tmp = -2 * c_tmp + norm_ij;
      c_tmp = ( c_tmp > 0) ? sqrt(c_tmp) : 0.0;
      }
      __syncthreads();

      if (j >= i) SM_dist[i * 2 * m + j] = c_tmp;
      if (j >= i) SM[i * 2 * m + j] = G_Id[leaf_id_g * ppl + block * m + j];
      
      if (j > i) SM_dist[j * 2 * m + i] = c_tmp;
      if (j > i) SM[j * 2 * m + i] = G_Id[leaf_id_g * ppl + block * m + i];
       
           
      __syncthreads();
           
      // TODO : should fix the issue for the initial value
      //int ind_knn = leaf_id_g * ppl * k_nn + (block * m + row_read) * k_nn + col_read;
      //SM_dist[row_read * 2 * m + col_read + m] = (col_read < k_nn) ? KNN_dist[ind_knn] : 1e30;
      //SM[row_read * 2 * m + col_read + m] = (col_read < k_nn) ? KNN_Id[ind_knn] :  0;
      
      SM_dist[i * 2 * m + j + m] = (j < k_nn) ? 1e30 : 1e30;
      SM[i * 2 * m + j + m] = (j < k_nn) ? 0 :  0;
      
      //int w = block * m + row_read;
      __syncthreads();
      

      // bitonic sort 

            

      for (int batch = 0; batch < 2; batch ++){
       
      int row = ind / (2 * m);
      int col = ind - row * 2 * m;

      //if (leaf_id_g == 1000 && w == 1000 ) printf("val = %.4f , id = %d \n", SM_dist[col], SM[col]);
      float tmp_f; 
      int tmp_i;
      int size = 2 *m;
      int col_tmp = (row + batch * m / 2) * 2 * m + col;
      for (int g = 2; g <= size; g *= 2){
        for (int l = g/2; l > 0; l /= 2){

          int ixj = col ^ l;
          int ixj_tmp = (row + batch * m / 2) * 2 * m + ixj;

          if (ixj > col){
            if(( col & g) == 0){
              if (SM_dist[col_tmp] > SM_dist[ixj_tmp]){
                
                tmp_f = SM_dist[ixj_tmp];
                SM_dist[ixj_tmp] = SM_dist[col_tmp];
                SM_dist[col_tmp] = tmp_f;
                
                tmp_i = SM[ixj_tmp];
                SM[ixj_tmp] = SM[col_tmp];
                SM[col_tmp] = tmp_i;
              }
           } else {
              if (SM_dist[col_tmp] < SM_dist[ixj_tmp]){
                
                tmp_f = SM_dist[ixj_tmp];
                SM_dist[ixj_tmp] = SM_dist[col_tmp];
                SM_dist[col_tmp] = tmp_f;
                
                tmp_i = SM[ixj_tmp];
                SM[ixj_tmp] = SM[col_tmp];
                SM[col_tmp] = tmp_i;
              }
           }
         }
       __syncthreads();
       }
     }

   } 
   
   int ind_knn = leaf_id_g * ppl * k_nn + (block * m + i) * k_nn + j;
   if (j < k_nn){
     KNN_dist[ind_knn] = SM_dist[ i * 2 * m + j];
     KNN_Id[ind_knn] = SM[i * 2 * m + j]; 

   }

}





__global__ void knn_kernel_sq(int* R, int* C, float* V, int* G_Id, float* Norms , int k_nn, float* KNN_dist, int* KNN_Id, int ppl, int max_nnz, int m, int blockInd) {

   // square partitions

    __shared__ int SM[4096];
    __shared__ float SM_dist[2048];

   int i = threadIdx.x;
   int j = threadIdx.y;

   //block = blockIdx.x;
   float tmp = -8 * blockInd + 4 * (m-1) * (m) - 7;
   int b_i = sqrt(tmp) / 2.0 - 0.5;
   b_i = m - 2 - b_i;
   int b_j = blockInd + b_i - (m-1) * (m)/2 + (m-1 - b_i) * (( m-1 - b_i) + 1)/2;
  
   //b_i++;
   b_j++; 
 
   int leaf_id_g = blockIdx.y;
   
   //if (leaf_id_g == 1000 && i == 0 && j == 0) printf("ind %d -> (%d, %d) \n", blockInd, b_i, b_j);

   

   int g_rowId_I = leaf_id_g * ppl + b_i * m + i;
   int g_rowId_J = leaf_id_g * ppl + b_j * m + j;

   int perm_i = G_Id[g_rowId_I];
   int perm_j = G_Id[g_rowId_J];

   int ind0_i = R[perm_i];
   int ind1_i = R[perm_i+1];

   int ind0_j = R[perm_j];
   int ind1_j = R[perm_j+1];

   int nnz_i = ind1_i - ind0_i;
   int nnz_j = ind1_j - ind0_j;

   
   
   float norm_ij = Norms[g_rowId_I] + Norms[g_rowId_J];

   int shift_i = max_nnz * i;
   //int shift_j = max_nnz * j;

   //__shared__ int SM_col[4096];

   for (int n_i = j; n_i< nnz_i; n_i += m) SM[shift_i + n_i] = C[ind0_i + n_i];
   //for (int n_j = i; n_j< nnz_j; n_j += m) SM_col[shift_j + n_j] = C[ind0_j + n_j];

   __syncthreads();

    
		float c_tmp = 0.0;
		int tmp_0, tmp_1, ind_jk, k, ret, testInd;

		ret = 0;
		testInd = 0;

		// loop over the elements of j
		for (int pos_k = 0; pos_k < nnz_j; pos_k++){
			//k = SM_col[max_nnz * j + pos_k];
      k = C[ind0_j + pos_k];
			// Binary search
			for (int l = nnz_i - ret; l > 1; l /= 2){
				tmp_0 = ret + l;
				tmp_1 = nnz_i - 1;
				testInd = (tmp_0 < tmp_1) ? tmp_0 : tmp_1;
				ret = (SM[testInd + shift_i] <= k) ? testInd : ret;
			}

			tmp_0 = ret + 1;
			tmp_1 = nnz_i - 1;
			testInd = (tmp_0 < tmp_1 ) ? tmp_0 : tmp_1;
			ret = (SM[testInd + shift_i] <= k) ? testInd : ret;
			ind_jk = (SM[ret + shift_i] == k) ? ret : -1;
			c_tmp += (ind_jk != -1) ? V[ind0_j + pos_k] * V[ind0_i + ind_jk] : 0;
    }
		c_tmp = -2 * c_tmp + norm_ij;
		c_tmp = ( c_tmp > 0) ? sqrt(c_tmp) : 0.0;

    __syncthreads();
      
    
    // horizontal merge

    SM_dist[i * 2*m + j] = c_tmp;
    SM[i * 2*m + j] =  G_Id[g_rowId_J];

    SM_dist[i * 2 * m + j + m] = (j < k_nn) ? KNN_dist[leaf_id_g * ppl * k_nn + (b_i * m + i) * k_nn + j] : 1e30;
    SM[i * 2 * m + j + m] = (j < k_nn) ? KNN_Id[leaf_id_g * ppl * k_nn + (b_i * m + i) * k_nn + j] : 0;

    
    // bitonic sort

    float tmp_f;
    int tmp_i;
    int size = 2 * m;
    int ind = i * m + j;
    int row = ind / (2 * m);
    int col = ind - row * 2 * m;
    for (int batch = 0; batch < 2; batch ++){
         
      int j_tmp = (row + batch * m /2)  * 2 * m + col;
      
      for (int g = 2; g <= size; g *= 2){
        for (int l = g/2; l > 0; l /= 2){

          int ixj = col ^ l;
          int ixj_tmp = (row + batch * m / 2) * 2*m + ixj;

          if (ixj > j){
            if(( j & g) == 0){
              if (SM_dist[j_tmp] > SM_dist[ixj_tmp]){

                tmp_f = SM_dist[ixj_tmp];
                SM_dist[ixj_tmp] = SM_dist[j_tmp];
                SM_dist[j_tmp] = tmp_f;

                tmp_i = SM[ixj_tmp];
                SM[ixj_tmp] = SM[j_tmp];
                SM[j_tmp] = tmp_i;
              }
           } else {
              if (SM_dist[j_tmp] < SM_dist[ixj_tmp]){

                tmp_f = SM_dist[ixj_tmp];
                SM_dist[ixj_tmp] = SM_dist[j_tmp];
                SM_dist[j_tmp] = tmp_f;

                tmp_i = SM[ixj_tmp];
                SM[ixj_tmp] = SM[j_tmp];
                SM[j_tmp] = tmp_i;
              }
           }
         }
       __syncthreads();
       }
     }
   
   if (col < k_nn){
     int ind_write = leaf_id_g * ppl * k_nn + (b_i * m + row + batch * m /2)* k_nn + col;
     KNN_dist[ind_write] = SM_dist[(row + batch * m /2) * 2 *m + col];
     KNN_Id[ind_write] = SM[(row + batch * m /2) * 2 *m + col];
     //if (leaf_id_g == 1000 && b_i == 30 && b_j == 30 && i == 31) printf(" val = %.4f , in = %d \n", KNN_dist[ind_write], j);
   } 
   

   }
   
   __syncthreads();
  
    /*
    // vertical merge

    SM_dist[j * 2*m + i] = c_tmp;
    SM[j * 2*m + i] =  G_Id[g_rowId_I];

    SM_dist[j * 2 * m + i + m] = (i < k_nn) ? KNN_dist[leaf_id_g * ppl * k_nn + (b_j * m + j) * k_nn + i] : 1e30;
    SM[j * 2 * m + i + m] = (i < k_nn) ? KNN_Id[leaf_id_g * ppl * k_nn + (b_j * m + j) * k_nn + i] : 0;

   
    // bitonic sort

    //float tmp_f;
    //int tmp_i;
    //int size = 2 * m;
    ind = j * m + i;
    row = ind / (2 * m);
    col = ind - row * 2 * m;
    for (int batch = 0; batch < 2; batch ++){
         
      int j_tmp = (row + batch * m /2)  * 2 * m + col;
      
      for (int g = 2; g <= size; g *= 2){
        for (int l = g/2; l > 0; l /= 2){

          int ixj = col ^ l;
          int ixj_tmp = (row + batch * m / 2) * 2*m + ixj;

          if (ixj > j){
            if(( j & g) == 0){
              if (SM_dist[j_tmp] > SM_dist[ixj_tmp]){

                tmp_f = SM_dist[ixj_tmp];
                SM_dist[ixj_tmp] = SM_dist[j_tmp];
                SM_dist[j_tmp] = tmp_f;

                tmp_i = SM[ixj_tmp];
                SM[ixj_tmp] = SM[j_tmp];
                SM[j_tmp] = tmp_i;
              }
           } else {
              if (SM_dist[j_tmp] < SM_dist[ixj_tmp]){

                tmp_f = SM_dist[ixj_tmp];
                SM_dist[ixj_tmp] = SM_dist[j_tmp];
                SM_dist[j_tmp] = tmp_f;

                tmp_i = SM[ixj_tmp];
                SM[ixj_tmp] = SM[j_tmp];
                SM[j_tmp] = tmp_i;
              }
           }
         }
       __syncthreads();
       }
     }
   
   if (col < k_nn){
     int ind_write = leaf_id_g * ppl * k_nn + (b_j * m + row + batch * m /2) * k_nn + col;
     KNN_dist[ind_write] = SM_dist[(row + batch * m /2) * 2 *m + col];
     KNN_Id[ind_write] = SM[(row + batch * m /2) * 2 *m + col];

   }


   }


   */



 
   
    
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
  //std::random_shuffle(&G_Id[0], &G_Id[M]);
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

  size_t free, total;
  cudaMemGetInfo(&free, &total);
  int log_size = log2(free / (sizeof(float)));
  double arr_len = pow(2, log_size); 

  int size_batch_leaves = arr_len / (ppl * ppl);
  if (size_batch_leaves > leaves) size_batch_leaves = leaves;
  int num_batch_leaves = (leaves + size_batch_leaves - 1) / size_batch_leaves;

  //printf("%d , %d  , %d \n", num_batch_I, num_batch_J, num_batch_leaves);
  float del_t1;
  cudaEvent_t t0; 
  cudaEvent_t t1;
  int blocks = m*m;
  
  int num_blocks_tri = ppl / m;
  dim3 dimBlock_tri(blocks, 1);	
  dim3 dimGrid_tri(num_blocks_tri, leaves); 
  
  int num_blocks_sq = m * (m-1) /2;
  dim3 dimBlock_sq(m, m);	
  dim3 dimGrid_sq(1, leaves); 
  
  dim3 dimBlock_norm(ppl);	
  dim3 dimGrid_norm(leaves); 
  
  float *d_Norms;
  
  printf("# leaves : %d \n", leaves);
  printf("# points/leaf : %d \n", ppl);
  printf("  max_nnz : %d \n", max_nnz); 
  printf(" block (tri) = (%d,%d) \n", blocks, 1);
  printf(" grid (tri) = (%d, %d) \n", num_blocks_tri, leaves);
  printf(" block (sq) = (%d,%d) \n", m, m);
  printf(" grid (sq) = (%d, %d) \n", 1, leaves);
  printf(" # points = %d \n" , M);
 
  checkCudaErrors(cudaMalloc((void **) &d_Norms, sizeof(float) * ppl * size_batch_leaves));


  checkCudaErrors(cudaEventCreate(&t0));
  checkCudaErrors(cudaEventCreate(&t1));

  checkCudaErrors(cudaEventRecord(t0, 0));
  compute_norm <<< dimGrid_norm, dimBlock_norm >>>(R, C, V, G_Id, d_Norms, ppl);
  
  for (int blockInd = 0; blockInd < num_blocks_sq; blockInd++){  
    checkCudaErrors(cudaDeviceSynchronize());
    if (blockInd == 0) knn_kernel_tri <<< dimGrid_tri, dimBlock_tri >>>(R, C, V, G_Id, d_Norms, k, knn, knn_Id, ppl, max_nnz, m);
    
    checkCudaErrors(cudaDeviceSynchronize());
    knn_kernel_sq <<< dimGrid_sq, dimBlock_sq >>>(R, C, V, G_Id, d_Norms, k, knn, knn_Id, ppl, max_nnz, m ,blockInd);
    checkCudaErrors(cudaDeviceSynchronize());
  } 
  
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaEventRecord(t1, 0));
  checkCudaErrors(cudaEventSynchronize(t1));
  checkCudaErrors(cudaEventElapsedTime(&del_t1, t0, t1));
 
  printf("\n Elapsed time (s) : %.4f \n ", del_t1/1000);
 
  checkCudaErrors(cudaFree(d_Norms));
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
    int M = 1024*2048;
    int leaves = 2048;
    d = 100000;
    int k = 32;
    nnzperrow = 8;
    int max_nnz = nnzperrow;
    int leaf_size = M / leaves; 
    

    bool print_pt = false;    
    bool print_res = true;    
    int test_leaf = 1000;    
    int test_pt = 1000;

    int *d_knn_Id, *h_knn_Id, *h_knn_Id_seq;
    float *d_knn, *h_knn, *h_knn_seq;

    h_R = (int *)malloc(sizeof(int)*(M+1));
    h_G_Id = (int *)malloc(sizeof(int)*(M));

    h_knn = (float *)malloc(sizeof(float) * M *k);
    h_knn_seq = (float *)malloc(sizeof(float) * M *k / leaves);
    h_knn_Id = (int *)malloc(sizeof(int) * M *k);
    h_knn_Id_seq = (int *)malloc(sizeof(int) * M *k / leaves);
    //memset(h_knn, 1000000.0, sizeof(float) * M * k);
    //memset(h_knn_Id, 0, sizeof(int) * M * k);
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
    //checkCudaErrors(cudaMemcpy(d_knn, h_knn, sizeof(float)*(M * k), cudaMemcpyHostToDevice));
    //checkCudaErrors(cudaMemcpy(d_knn_Id, h_knn_Id, sizeof(int)*(M * k), cudaMemcpyHostToDevice));
    //checkCudaErrors(cudaMemset(d_knn, 1000, sizeof(float) * M * k));  
    //checkCudaErrors(cudaMemset(d_knn_Id, 0, sizeof(int) * M * k));  

    printf("Random csr is generated  \n");

    gpu_knn(d_R, d_C, d_V, d_G_Id, M, leaves, k, d_knn, d_knn_Id, max_nnz);
    
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
      match = (h_knn_Id_seq[test_pt*k + i] == h_knn_Id[ind]);
      if (print_res){
      printf("seq ind %d,\t gpu_ind %d , \t match %d , \t v_seq %.4f, \t v_gpu %.4f , \t ind = %d\n", h_knn_Id_seq[test_pt*k + i], h_knn_Id[ind], match, h_knn_seq[test_pt*k + i], h_knn[ind], ind);
      }
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
