
#include <stdio.h> 
#include <stdlib.h>
//#include <cusparse.h>
#include <cuda.h>
#include <helper_cuda.h>
#include <cuda_runtime.h>
#include <algorithm>






__global__ void compute_dist(int* R, int* C, float* V, float* K, int* K_ID, int m_i, int m_j , int k_nn, int M_I){
    int i = threadIdx.x ; 

    int col_Id = threadIdx.x;
    //int row_Id = blockIdx.x;
    int row_Id = threadIdx.y;
    
    //int blockId_J = threadIdx.y;
    int blockId_J = blockIdx.x;
    int blockId_I = blockIdx.y;
   
   
    int leaf_id_g = threadIdx.z +blockIdx.z * blockDim.z; 
    //int leaf_id_g = blockIdx.z; 

    //int binsearch_thread = threadIdx.z;
    //int num_thread = blockDim.z; 

    int g_rowId_I = leaf_id_g * M_I + blockId_I * m_i + row_Id;
    int g_rowId_J = leaf_id_g * M_I + blockId_J * m_j + col_Id;
     
    int ind0_i = R[g_rowId_I];
    int ind1_i = R[g_rowId_I + 1];

    int ind0_j = R[g_rowId_J];
    int ind1_j = R[g_rowId_J + 1];
 
    int nnz_i = ind1_i - ind0_i;
    int nnz_j = ind1_j - ind0_j;


    float norm_ij = 0;

    

    //__shared__ int sj[512];
    //__shared__ float vj[512];
    
    int si[512];
    float vi[512];
    int sj[512];
    float vj[512];
   
    for (int n_i = 0; n_i < nnz_i; n_i++){
      si[n_i] = C[ind0_i + n_i];
      vi[n_i] = V[ind0_i + n_i];
      norm_ij += vi[n_i]*vi[n_i];
    }
    for (int n_j = 0; n_j < nnz_j; n_j++){
      sj[n_j] = C[ind0_j + n_j];
      vj[n_j] = V[ind0_j + n_j];
      norm_ij += vj[n_j]*vj[n_j];
    }
    
 
     
    float c_tmp = 0;
    float c;
    int log_max_nnz = 5;
    int nnz_j_tmp = nnz_j;
    int tmp_0, tmp_1, ind_jk, k, ret, testInd; 
    int ind0, ind1;
    
    ret=0; 
    testInd = 0;

    for (int pos_k=0; pos_k<nnz_i;pos_k++){       
        k = si[pos_k];
        //ret = testInd;
           
        // Binary search 
        for (int l=nnz_j-ret; l > 1; l/=2){
            tmp_0 = ret+l;
            tmp_1 = nnz_j-1;
            testInd = (tmp_0 < tmp_1) ? tmp_0: tmp_1;
            ret = (sj[testInd] <= k) ? testInd : ret ;
        }
        tmp_0 = ret+1;
        tmp_1 = nnz_j-1;
        testInd = (tmp_0 < tmp_1) ? tmp_0: tmp_1;
        ret = (sj[testInd] <= k) ? testInd : ret;
        ind_jk = (sj[ret] == k) ? ret : -1;
        c_tmp += (ind_jk != -1) ? vi[pos_k]*vj[ind_jk] : 0;
        
    }
    /*
    for (int pos_k=0; pos_k<nnz_i;pos_k++){       
        k = si[pos_k];
        
        ret=0; 
        testInd = 0;
        
        // Binary search 
        for (int l=nnz_j; l > 1; l/=2){
            tmp_0 = ret+l;
            tmp_1 = nnz_j-1;
            testInd = (tmp_0 < tmp_1) ? tmp_0: tmp_1;
            ret = (sj[testInd] <= k) ? testInd : ret ;
        }
        tmp_0 = ret+1;
        tmp_1 = nnz_j-1;
        testInd = (tmp_0 < tmp_1) ? tmp_0: tmp_1;
        ret = (sj[testInd] <= k) ? testInd : ret;
        ind_jk = (sj[ret] == k) ? ret : -1;
        c_tmp += (ind_jk != -1) ? vi[pos_k]*vj[ind_jk] : 0;
        
    }
    */
    c = -2*c_tmp + norm_ij;
    c_tmp = (c > 0) ? c : 0.0;
    c_tmp = sqrt(c_tmp);

    // bitonic sort 
    __shared__ float kvals[2048];
    __shared__ int id_k[2048];

    int ind_s = threadIdx.y * blockDim.x + threadIdx.x; 
    kvals[ind_s] = c_tmp; 
    id_k[ind_s] = g_rowId_J;
    
    
    __syncthreads();
    
    int log_size = 0;
    int m_j_tmp = m_j;
    while (m_j_tmp >>= 1) ++log_size;

    //int log_size = log2(m_j);
    int size = (pow(2,log_size) < m_j) ? pow(2, log_size+1) : m_j;
    // bitonic sort  
    float tmp_f;
    int tmp_i;
    //if (g_rowId_I == 13) printf("col = %d , c_tmp = %.2f , ind_s = %d , m_j = %d , m_i = %d \n", g_rowId_J, c_tmp, ind_s, threadIdx.x, threadIdx.y); 
    for (int g = 2; g <= size; g *= 2){
      for (int l = g/2; l>0; l /= 2){
	    int ixj = i ^ l;
      int ixj_tmp = threadIdx.y * blockDim.x + ixj;
	    if (ixj > i){
		    if ((i & g) == 0){
			    if (kvals[ind_s] > kvals[ixj_tmp]){ 
               tmp_f = kvals[ixj_tmp]; 
               kvals[ixj_tmp] = kvals[ind_s]; 
               kvals[ind_s] = tmp_f;
               tmp_i = id_k[ixj_tmp]; 
               id_k[ixj_tmp] = id_k[ind_s]; 
               id_k[ind_s] = tmp_i;
                }
		    } else {
			    if (kvals[ind_s] < kvals[ixj_tmp]){ 
               tmp_f = kvals[ixj_tmp]; 
               kvals[ixj_tmp] = kvals[ind_s]; 
               kvals[ind_s] = tmp_f;
               tmp_i = id_k[ixj_tmp]; 
               id_k[ixj_tmp] = id_k[ind_s]; 
               id_k[ind_s] = tmp_i;
                } 
		    }
	      }
	    __syncthreads();
      }
    }
     
    if (col_Id < k_nn){
	    int col_write = blockId_J * k_nn + col_Id; 
	    int row_write = blockId_I * m_i + row_Id;
	    int ind_write = leaf_id_g * M_I * k_nn + row_write * k_nn + col_write; 
      //printf("leaf_id = %d , row_write = %d , col_write = %d , ind_write = %d \n", leaf_id_g, row_write, col_write , ind_write);
	    K[ind_write] = kvals[ind_s];
	    K_ID[ind_write] = id_k[ind_s];
    }
}


__global__ void find_neighbor(float* knn, int* knn_Id, float* K, int* K_Id, int k, int M_I){

    int col_Id = threadIdx.x;
    
    int row_Id = threadIdx.y;

    //int blockId_J = threadIdx.y;
    int blockId_J = blockIdx.x;
    int blockId_I = blockIdx.y;


    int leaf_id_g = threadIdx.z +blockIdx.z * blockDim.z;

    __shared__ float Dist[2048];
    __shared__ int Dist_Id[2048];

		int col_write = blockId_J * k + col_Id - k;
		int row_write = blockId_I * blockDim.y + row_Id;
		int ind_read = leaf_id_g * M_I * k + row_write * k + col_write;
    int ind_shared = row_Id * blockDim.x + col_Id;
    int ind_knn = leaf_id_g * M_I * k + blockId_I * blockDim.y * k + row_Id * k + col_Id;

    int size = blockDim.x + k;
    int true_size = size;
    //while (tmp >>= 1) ++log_size;
    size = size - 1;
    size |= size >> 1;    
    size |= size >> 2;    
    size |= size >> 4;    
    size |= size >> 8;    
    size |= size >> 16;    
    //size |= size >> 1;
    size++;    


    int i = col_Id;

    //Dist[ind_shared] = (col_Id < k) ? knn[ind_knn] : (col_Id < size) ? K[ind_read] : 1e30;
    Dist[ind_shared] = (col_Id < k) ? 1e30 : (col_Id < size) ? K[ind_read] : 1e30;
    //Dist_Id[ind_shared] = (col_Id < k) ? knn_Id[ind_knn] : (col_Id < size) ? K_Id[ind_read] : 0;
    Dist_Id[ind_shared] = (col_Id < k) ? 0 : (col_Id < size) ? K_Id[ind_read] : 0;
 
    __syncthreads();

    int diff = size - true_size;
    // bitonic sort
    float tmp_f;
    int tmp_i;
    for (int g = 2; g <= size; g *= 2){
      for (int l = g/2; l>0; l /= 2){
      int ixj = i ^ l;
      int ixj_tmp = row_Id * blockDim.x + ixj;
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
    if (col_Id < k){
      col_write = blockId_I * k + col_Id;
      row_write = blockId_I * blockDim.y + row_Id;
      int ind_write = leaf_id_g * M_I * k + row_write * k + col_write;
      knn[ind_write] = Dist[ind_shared];
      knn_Id[ind_write] = Dist_Id[ind_shared];
    } 
    
}





void gen_sparse(int M, int nnzperrow, int d, int *R, int *C, float *V) {
 
    R[0] = 0;
    for (int i=0; i < M; i++){
      R[i+1] = R[i] + nnzperrow;
      for (int j=0; j < nnzperrow; j++){
          int ind = R[i]+j;
          int val = (j)*d/nnzperrow;
          //val = rand()%d;
          C[ind] = val;
          V[ind] = rand()%10;
        }    
        std::sort(C+i*nnzperrow, C+(i+1)*nnzperrow);
    }
    R[M] = nnzperrow*M;
}


int main(int argc, char **argv)
{
  /*
  float t_1, t_2, t_3, wtime;



  t_1 = 0;
  t_2 = 0;
  t_3 = 0;
  wtime = 0;
  */

   float del_t1, del_t2;
  //, del_t2, del_t3;

    checkCudaErrors(cudaSetDevice(0));

    int d, nnzperrow;
    float *h_V, *d_V;
    int *h_C, *d_C;
    int *h_R, *d_R;
    float *d_K;
    int *d_K_ID;
    int nnz_r;
    cudaEvent_t t0; 
    cudaEvent_t t1;
    cudaEvent_t t2;

    int M = 1024*2048;
    int leaves = 2048;
    d = 1000;
    int k = 32;
    nnzperrow = 16;
    
    int pointsperleaf = M/leaves;
    int m_j = min(256, pointsperleaf);
    int m_i = 4;
    int blocksize_leaf = 1;
    
    int size_batch_I = (pointsperleaf)/m_i;
    int size_batch_J = (pointsperleaf + m_j - 1) / m_j;
    int M_I = M/leaves;
    int batch_leaves = leaves / blocksize_leaf;
    //int batch_leaves = 1;

    int *d_knn_Id;
    float *d_knn;

    h_V = (float *)malloc(sizeof(float)*nnzperrow*M);
    h_C = (int *)malloc(sizeof(int)*nnzperrow*M);
    h_R = (int *)malloc(sizeof(int)*(M+1));

    // generate random data 

    gen_sparse(M, nnzperrow, d , h_R, h_C, h_V);   
    /*
    for (int i = 0; i < M; i++){
        for (int j = 0; j < nnzperrow; j++)
        printf("row = %d , col_ind = %d , val = %.4f \n", i , h_C[h_R[i]+j], h_V[h_R[i]+j]);
    }    
    */
    checkCudaErrors(cudaMalloc((void **) &d_K, sizeof(int)*M*k));
    checkCudaErrors(cudaMalloc((void **) &d_K_ID, sizeof(int)*M*k));
    checkCudaErrors(cudaMalloc((void **) &d_R, sizeof(int)*(M+1)));
    checkCudaErrors(cudaMalloc((void **) &d_C, sizeof(int)*nnzperrow*M));
    checkCudaErrors(cudaMalloc((void **) &d_V, sizeof(float)*nnzperrow*M));
    checkCudaErrors(cudaMalloc((void **) &d_knn_Id, sizeof(int)*M*k));
    checkCudaErrors(cudaMalloc((void **) &d_knn, sizeof(float)*M*k));

    //checkCudaErrors(cudaMemset(d_knn_Id, 0, sizeof(int)*M*k)); 
    //checkCudaErrors(cudaMemset(d_knn, 1000.0, sizeof(float)));
 
    checkCudaErrors(cudaMemcpy(d_C, h_C, sizeof(int)*nnzperrow*M, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_V, h_V, sizeof(float)*nnzperrow*M, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_R, h_R, sizeof(int)*(M+1), cudaMemcpyHostToDevice));
    dim3 dimBlock(m_j, m_i, blocksize_leaf);	// 512 threads per thread-block
    dim3 dimGrid(size_batch_J, size_batch_I, batch_leaves); // Enough thread-blocks to cover N
    dim3 dimBlock_n(k*(size_batch_J+1), m_i, blocksize_leaf);
    dim3 dimGrid_n(1 , size_batch_I, batch_leaves);

    printf("m_j = %d , size_batch_J = %d , m_i = %d , size_batch_I = %d , blocksize_leaf = %d , batch_leavs = %d \n", 
           m_j , size_batch_J , m_i , size_batch_I , blocksize_leaf, batch_leaves);
    checkCudaErrors(cudaEventCreate(&t0));
    checkCudaErrors(cudaEventCreate(&t1));
    checkCudaErrors(cudaEventCreate(&t2));

    checkCudaErrors(cudaEventRecord(t0, 0));
    compute_dist <<< dimGrid, dimBlock >>>(d_R, d_C, d_V, d_K, d_K_ID, m_i, m_j, k, M_I);
    checkCudaErrors(cudaDeviceSynchronize());
    
    checkCudaErrors(cudaEventRecord(t1, 0));
    checkCudaErrors(cudaEventSynchronize(t1));
    checkCudaErrors(cudaEventElapsedTime(&del_t1, t0, t1));

    find_neighbor <<< dimGrid_n, dimBlock_n >>>(d_knn, d_knn_Id, d_K, d_K_ID, k, M_I);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaEventRecord(t2, 0));
    checkCudaErrors(cudaEventSynchronize(t2));
    checkCudaErrors(cudaEventElapsedTime(&del_t2, t1, t2));
    

    printf("\n Elapsed time (ms) : %.4f \n ", del_t1);
    printf("\n Elapsed time (ms) : %.4f \n ", del_t2);


    
    printf("\n\n");
    checkCudaErrors(cudaFree(d_R));
    checkCudaErrors(cudaFree(d_C));
    checkCudaErrors(cudaFree(d_V));
    checkCudaErrors(cudaFree(d_K));
    checkCudaErrors(cudaFree(d_K_ID));
    checkCudaErrors(cudaEventDestroy(t0));
    checkCudaErrors(cudaEventDestroy(t1));
    free(h_R);
    free(h_C);
    free(h_V);


}
