
#include <stdio.h> 
#include <stdlib.h>
//#include <cusparse.h>
#include <cuda.h>
#include <helper_cuda.h>
#include <cuda_runtime.h>
#include <algorithm>






__global__ void compute_dist(int* R, int* C, float* V, float* K, int m_i, int m_j , int k_nn, int M_I, int leaf_id_g, int max_nnz){

    int col_Id = threadIdx.x;
    //int row_Id = blockIdx.x;
    int row_Id = threadIdx.y;
    
    //int blockId_J = threadIdx.y;
    int blockId_J = blockIdx.x;
    int blockId_I = blockIdx.y;
   
   
    //int leaf_id_g = threadIdx.z +blockIdx.z * blockDim.z; 
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
    
    //int si[256];
    __shared__ int si[4096];
    //float vi[256];
    //int sj[256];
    //float vj[256];
    int shift = max_nnz*threadIdx.y;
    //if (leaf_id_g == 0) printf("Id_y = %d , shift = %d \n", threadIdx.y , shift);
    
    for (int n_i = 0; n_i < nnz_i; n_i++){
      si[n_i + shift] = C[ind0_i + n_i];
      //vi[n_i] = V[ind0_i + n_i];
      //norm_ij += vi[n_i]*vi[n_i];
      norm_ij += V[ind0_i + n_i]*V[ind0_i + n_i];
    }
    for (int n_j = 0; n_j < nnz_j; n_j++){
      //sj[n_j] = C[ind0_j + n_j];
      //vj[n_j] = V[ind0_j + n_j];
      //norm_ij += vj[n_j]*vj[n_j];
      norm_ij += V[ind0_j + n_j]*V[ind0_j + n_j];
    }
     
     
     
    float c_tmp = 0;
    float c;
    
    int tmp_0, tmp_1, ind_jk, k, ret, testInd; 
    
    ret=0; 
    testInd = 0;

    
    
    for (int pos_k=0; pos_k<nnz_j;pos_k++){       
        //k = sj[pos_k];
        k = C[ind0_j + pos_k];
        //ret = testInd;
           
        // Binary search 
        for (int l=nnz_i-ret; l > 1; l/=2){
            tmp_0 = ret+l;
            tmp_1 = nnz_i-1;
            testInd = (tmp_0 < tmp_1) ? tmp_0: tmp_1;
            ret = (si[testInd+ shift] <= k) ? testInd : ret ;
        }
        tmp_0 = ret+1;
        tmp_1 = nnz_i-1;
        testInd = (tmp_0 < tmp_1) ? tmp_0: tmp_1;
        ret = (si[testInd+ shift] <= k) ? testInd : ret;
        ind_jk = (si[ret+ shift] == k) ? ret : -1;
        c_tmp += (ind_jk != -1) ? V[ind0_j + pos_k]*V[ind0_i + ind_jk] : 0;
        
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
        
        //c_tmp += (ind_jk != -1) ? vi[pos_k]*vj[pos_k] : 0;
        //c_tmp +=  vi[pos_k]*vi[pos_k] ;
        
    }
    */
    c = -2*c_tmp + norm_ij;
    c_tmp = (c > 0) ? c : 0.0;
    c_tmp = sqrt(c_tmp);
    
	  int col_write = blockId_J * m_j + col_Id; 
	  int row_write = blockId_I * m_i + row_Id;
	  int ind_write = row_write * M_I + col_write;
    K[ind_write] = c_tmp;
    //if (leaf_id_g == 2 && row_write == 0) printf("leaf = %d , row = %d , col = %d , val = %.2f \n", leaf_id_g, row_write, col_write , c_tmp);

    /*
    // bitonic sort 
    __shared__ float kvals[2048];
    //__shared__ int id_k[2048];

    int ind_s = threadIdx.y * blockDim.x + threadIdx.x; 
    kvals[ind_s] = c_tmp; 
    //id_k[ind_s] = g_rowId_J;
    si[ind_s] = g_rowId_J;
    
    
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
               //tmp_i = id_k[ixj_tmp]; 
               tmp_i = si[ixj_tmp]; 
               si[ixj_tmp] = si[ind_s]; 
               //id_k[ixj_tmp] = id_k[ind_s]; 
               si[ind_s] = tmp_i;
               //id_k[ind_s] = tmp_i;
                }
		    } else {
			    if (kvals[ind_s] < kvals[ixj_tmp]){ 
               tmp_f = kvals[ixj_tmp]; 
               kvals[ixj_tmp] = kvals[ind_s]; 
               kvals[ind_s] = tmp_f;
               //tmp_i = id_k[ixj_tmp]; 
               //id_k[ixj_tmp] = id_k[ind_s]; 
               //id_k[ind_s] = tmp_i;
               tmp_i = si[ixj_tmp]; 
               si[ixj_tmp] = si[ind_s]; 
               si[ind_s] = tmp_i;
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
	    //K_ID[ind_write] = id_k[ind_s];
	    K_ID[ind_write] = si[ind_s];
    }
    */ 
    
}


//__global__ void find_neighbor(float* knn, int* knn_Id, float* K, int* K_Id, int k, int M_I){
__global__ void find_neighbor(float* knn, int* knn_Id, float* K, int k, int M_I, int m_j, int leaf_id_g){

    int col_Id = threadIdx.x;
    
    int row_Id = blockIdx.x;

    //int blockId_J = threadIdx.y;
    //int blockId_J = blockIdx.x;
    //int blockId_I = blockIdx.y;
    

    //int leaf_id_g = threadIdx.z +blockIdx.z * blockDim.z;

    __shared__ float Dist[2048];
    __shared__ int Dist_Id[2048];

		//int col_write = blockId_J * m_j + col_Id;
		//int row_write = blockId_I * blockDim.y + row_Id;
		//int ind_read = leaf_id_g * M_I * k + row_write * k + col_write;
		//int ind_read = row_write * M_I + col_write;
		int ind_read = row_Id * M_I + col_Id;
    //int ind_shared = row_Id * blockDim.x + col_Id;
    //int ind_knn = leaf_id_g * M_I * k + blockId_I * blockDim.y * k + row_Id * k + col_Id;
    
    int size = blockDim.x;
    
    
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
    //if (leaf_id_g == 2047) printf("row = %d , col = %d 0 , ind_shared = %d \n", row_write, col_write, ind_shared);
    //Dist[ind_shared] = (col_Id < k) ? knn[ind_knn] : (col_Id < size) ? K[ind_read] : 1e30;
    Dist[col_Id] = K[ind_read];
    //Dist_Id[ind_shared] = (col_Id < k) ? knn_Id[ind_knn] : (col_Id < size) ? K_Id[ind_read] : 0;
    //Dist_Id[ind_shared] = (col_Id < k) ? 0 : (col_Id < size) ? K_Id[ind_read] : 0;
    Dist_Id[col_Id] = col_Id;
    int ind_shared = col_Id;
    //if (leaf_id_g == 2047) printf("row = %d , col = %d 1 , ind_shared = %d , \n", row_write, col_write, ind_shared);
     
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
      //if (leaf_id_g == 0 && row_write == 0) printf("i = %d , ixj = %d , Dist[%d] = %.2f \n", i , ixj, i, Dist[i]);
      }
    }
    //if (leaf_id_g == 2 && row_write == 0) printf("i = %d , col = %d , Dist[%d] = %.2f \n", i , col_Id, i, Dist[i]);
    size = 2*k;
    
    //if (col_Id >= k && cold_Id < size) Dist[ind_shared] = knn[ind_knn];
    if (col_Id >= k && col_Id < size) Dist[col_Id] = 1e30;
    if (col_Id >= k && col_Id < size) Dist_Id[col_Id] = 0;

    // merge with knn

    for (int g = 2; g <= size; g *= 2){
      for (int l = g/2; l>0; l /= 2){
      int ixj = i ^ l;
      int ixj_tmp =  ixj;
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
    


    //if (leaf_id_g == 2 && row_write == 0) printf("i = %d , col = %d , Dist[%d] = %.2f \n", i , col_Id, i, Dist[i]);
    if (col_Id < k){
      //col_write = row_Id * k + col_Id;
      //row_write = blockId_I * blockDim.y + row_Id;
      int ind_write = leaf_id_g * M_I * k + row_Id * k + col_Id;
      knn[ind_write] = Dist[col_Id];
      knn_Id[ind_write] = Dist_Id[col_Id];
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
          V[ind] = rand()%100;
        }    
      std::sort(C+R[i], C+(R[i+1]));
    }
}

void gen_R(int M, int nnzperrow, int *R, int d) {  
  R[0] = 0;
  int tot_nnz = 0;
  int val;
  for (int m =1; m <= M; m++){ 
   //val = nnzperrow/2 + rand()%nnzperrow;
   val = nnzperrow; //+ rand()%nnzperrow;
   if (val > d) val = 1; 
   tot_nnz += val;
   R[m] = tot_nnz;
  } 
  
}

void gpu_knn(int *R, int *C, float *V, int M, int leaves, int k, float *knn, int *knn_Id){

  

	int pointsperleaf = M/leaves;
	int m_j = min(64, pointsperleaf);
	int m_i = 16;
	int max_nnz = 200;
	int size_batch_I = (pointsperleaf)/m_i;
	int size_batch_J = (pointsperleaf + m_j - 1) / m_j;
	int M_I = M/leaves;
  float del_t1;
  cudaEvent_t t0; 
  cudaEvent_t t1;
  
  dim3 dimBlock(m_j, m_i);	
  dim3 dimGrid(size_batch_J, size_batch_I); 
  dim3 dimBlock_n(M_I);
  dim3 dimGrid_n(M_I);


  float *d_K;
  checkCudaErrors(cudaMalloc((void **) &d_K, sizeof(int)*pointsperleaf*pointsperleaf));


  checkCudaErrors(cudaEventCreate(&t0));
  checkCudaErrors(cudaEventCreate(&t1));

  checkCudaErrors(cudaEventRecord(t0, 0));
  for (int leaf_id_g = 0; leaf_id_g < leaves; leaf_id_g++){
    compute_dist <<< dimGrid, dimBlock >>>(R, C, V, d_K, m_i, m_j, k, M_I, leaf_id_g, max_nnz);
    checkCudaErrors(cudaDeviceSynchronize());
    find_neighbor <<< dimGrid_n, dimBlock_n >>>(knn, knn_Id, d_K, k, M_I, m_j, leaf_id_g);
  } 
  
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaEventRecord(t1, 0));
  checkCudaErrors(cudaEventSynchronize(t1));
  checkCudaErrors(cudaEventElapsedTime(&del_t1, t0, t1));
  printf("\n Elapsed time (ms) : %.4f \n ", del_t1);

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

    int M = 1024*2048;
    int leaves = 2048;
    d = 10000;
    int k = 32;
    nnzperrow = 128;
    
    
    

    int *d_knn_Id;
    float *d_knn;

    h_R = (int *)malloc(sizeof(int)*(M+1));

    // generate random data 
    gen_R(M, nnzperrow, h_R, d);
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
    checkCudaErrors(cudaMalloc((void **) &d_C, sizeof(int)*tot_nnz));
    checkCudaErrors(cudaMalloc((void **) &d_V, sizeof(float)*tot_nnz));
    checkCudaErrors(cudaMalloc((void **) &d_knn_Id, sizeof(int)*M*k));
    checkCudaErrors(cudaMalloc((void **) &d_knn, sizeof(float)*M*k));
 
    checkCudaErrors(cudaMemcpy(d_C, h_C, sizeof(int)*tot_nnz, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_V, h_V, sizeof(float)*tot_nnz, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_R, h_R, sizeof(int)*(M+1), cudaMemcpyHostToDevice));

    printf("Gen rnd mat finished \n");
    //printf("m_j = %d , size_batch_J = %d , m_i = %d , size_batch_I = %d , blocksize_leaf = %d , batch_leavs = %d \n", 
           //m_j , size_batch_J , m_i , size_batch_I , blocksize_leaf, batch_leaves);

    gpu_knn(d_R, d_C, d_V, M, leaves, k, d_knn, d_knn_Id);
    
    printf("\n\n");
    checkCudaErrors(cudaFree(d_R));
    checkCudaErrors(cudaFree(d_C));
    checkCudaErrors(cudaFree(d_V));
    free(h_R);
    free(h_C);
    free(h_V);


}
