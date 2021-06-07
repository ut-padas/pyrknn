
#include <stdio.h> 
#include <stdlib.h>
//#include <cusparse.h>
#include <cuda.h>
#include <helper_cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cuda_profiler_api.h>

#define shared_size  8192
#define MaxProcperBlock  1024 




__global__ void compute_norm(int* R, int* C, float* V, int* G_Id, float* K, int M_I, int leaf_batch_g , int k){

  int row = threadIdx.x;
  int leaf_id_g = leaf_batch_g * gridDim.x + blockIdx.x;

  int g_rowId = leaf_id_g * M_I + row;

  int g_Id = G_Id[g_rowId];
  int ind0_i = R[g_Id];

  int nnz = R[g_Id + 1] - ind0_i;
  float norm_i = 0.0;

  for (int n_i = 0; n_i < nnz; n_i++) norm_i += V[ind0_i + n_i] * V[ind0_i + n_i];  
  int ind_write = blockIdx.x * M_I + row * k;
  K[ind_write] = norm_i;

}



__global__ void compute_dist(int* R, int* C, float* V, int* G_Id,  float* K, int* K_Id, int m, int k_nn, int M_I, int leaf_batch_g, int max_nnz, int M){

    int j_Id = threadIdx.x;
    int blockId_j = blockIdx.x;

    int row_Id_leaf = blockIdx.y;
       
    int leaf_id_g = leaf_batch_g * gridDim.z + blockIdx.z;
    
    int g_rowId_I = leaf_id_g * M_I + row_Id_leaf;
    
    if (g_rowId_I >= M) return;
    
    int g_Id_i = G_Id[g_rowId_I]; 
   

    int ind0_i = R[g_Id_i];
    int ind1_i = R[g_Id_i + 1];

 
    int nnz_i = ind1_i - ind0_i;


    int ind_read_norm_i = blockIdx.z * M_I + row_Id_leaf * k_nn;
    float norm_i = K[ind_read_norm_i];
    //if (row_Id_leaf == j_Id) printf("row = %d, norm = %.4f\n", row_Id_leaf, norm_i);
    __shared__ int si[1024];
    __shared__ float dist[1024];
    __shared__ int dist_Id[1024];
    __shared__ float c_tmp[1];

    c_tmp[1] = 0.0;
    for (int n_i = j_Id; n_i < nnz_i; n_i += blockDim.x) si[n_i] = C[ind0_i + n_i];
     
    __syncthreads();
    
    int k; 
    dist_Id[j_Id] = blockId_j * blockDim.x + j_Id;
   
    // for loop over the points    
    for (int col_Id = 0; col_Id < blockDim.x; col_Id++) {
       
      int col_Id_leaf = blockId_j * blockDim.x + col_Id;
      int g_Id_J = leaf_id_g * M_I + col_Id_leaf;
      //printf("row = %d , col = %d \n", row_Id_leaf, col_Id_leaf); 
      int g_Id_j = G_Id[g_Id_J]; 
      int ind0_j = R[g_Id_j];
      int ind1_j = R[g_Id_j + 1];
      
      int nnz_j = ind1_j - ind0_j;
      int ind_read_norm_j = blockIdx.z * M_I + col_Id_leaf*k_nn; 
      float norm_ij = norm_i + K[ind_read_norm_j];
      c_tmp[0] = 0.0;   
      // for loop over the elements
      for (int pos_k =0; pos_k < nnz_i; pos_k++){
      if (j_Id < nnz_i) {
          k = C[ind0_j + pos_k];
          if (k == si[j_Id]) c_tmp[0] += V[ind0_j + pos_k] * V[ind0_i + j_Id];
          //if (row_Id_leaf == col_Id_leaf) printf("r,c = (%d,%d) , c_i = %d, c_j = %d , c_tmp =%.4f \n", row_Id_leaf, col_Id_leaf, si[j_Id],k, c_tmp[0]); 
      }
      __syncthreads(); 
    
      }
      c_tmp[0] = -2*c_tmp[0] + norm_ij;
      //c_tmp = (c_tmp > 0) ? sqrt(c_tmp) : 0.0;
      if (col_Id == j_Id) dist[col_Id] = c_tmp[0];
      //if (row_Id_leaf == col_Id_leaf && j_Id == col_Id) printf("row,col = (%d,%d), val = %.4f , norms = %.4f\n", row_Id_leaf, col_Id_leaf, c_tmp[0], norm_ij); 
         
    }

    // bitonic sort 
    //__shared__ int id_k[2048];

     

    int size = blockDim.x;
    // bitonic sort  
    float tmp_f;
    int tmp_i;
    
    for (int g = 2; g <= size; g *= 2){
      for (int l = g/2; l>0; l /= 2){
	    int ixj = j_Id ^ l;
	    if (ixj > j_Id){
		    if ((j_Id & g) == 0){
			    if (dist[j_Id] > dist[ixj]){ 
               tmp_f = dist[ixj]; 
               dist[ixj] = dist[j_Id]; 
               dist[j_Id] = tmp_f;
    
               tmp_i = dist_Id[ixj]; 
               dist_Id[ixj] = dist_Id[j_Id]; 
               dist_Id[j_Id] = tmp_i;
                }
		    } else {
			    if (dist[j_Id] < dist[ixj]){ 
               tmp_f = dist[ixj]; 
               dist[ixj] = dist[j_Id]; 
               dist[j_Id] = tmp_f;
    
               tmp_i = dist_Id[ixj]; 
               dist_Id[ixj] = dist_Id[j_Id]; 
               dist_Id[j_Id] = tmp_i;
               } 
		    }
	      }
	    __syncthreads();
      }
    }
    if (j_Id < k_nn) {
	    int col_write = blockIdx.x * k_nn + j_Id; 
	    int row_write = row_Id_leaf;
	    int ind_write = leaf_id_g * M_I * k_nn + row_write * k_nn + col_write;
       
	    K[ind_write] = dist[j_Id];
	    K_Id[ind_write] = dist_Id[j_Id];
    }
    
}

__global__ void find_neighbor(float* knn, int* knn_Id, float* K, int* K_Id, int* G_Id, int k, int M_I, int leaf_batch_g, int M){

    int col_Id = threadIdx.x; 
    int row_Id = blockIdx.x;

    if (row_Id >= M || col_Id >= M) return;
 
    __shared__ float Dist[4096];
    __shared__ int Dist_Id[4096];

    int size = blockDim.x;
    int leaf_id_g = leaf_batch_g * gridDim.y + blockIdx.y;
    
    int ind_K = blockIdx.z * M_I * M_I + row_Id * M_I + col_Id; 
    int i = col_Id;
    
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

  int ind_knn = leaf_id_g * M_I * k + row_Id * k + col_Id;

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



__global__ void merge(float* knn, int* knn_Id, float* K, int* K_Id, int k, int M_I, int m_j, int leaf_id_g){

  int col_Id = threadIdx.x; 
  int row_Id = blockIdx.x;
  //int leaf_id_g = threadIdx.z +blockIdx.z * blockDim.z;

  __shared__ float Dist[2048];
  __shared__ int Dist_Id[2048];

  int size = 2*k;


  int ind_knn = leaf_id_g * M_I * k + row_Id * k + col_Id;
  int ind_K = ind_knn - k;

  //Dist[col_Id] = (col_Id < k) ? knn[ind_knn] : K[ind_K]; 
  //Dist_Id[col_Id] = (col_Id < k) ? knn_Id[ind_knn] : K_Id[ind_K]; 

  Dist[col_Id] = (col_Id < k) ? 1e30 : K[ind_K]; 
  Dist_Id[col_Id] = (col_Id < k) ? 0 : K_Id[ind_K]; 


  // merge with knn
  int i = col_Id;
  float tmp_f;
  int tmp_i;  
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
    }
    
    if (col_Id < k){
      knn[ind_knn] = Dist[col_Id];
      knn_Id[ind_knn] = Dist_Id[col_Id];
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
   val = 1 + rand()%(2*nnzperrow);
   //val = nnzperrow; //+ rand()%nnzperrow;
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
 
	int pointsperleaf = M/leaves;
	 
  int size_batch_leaves = (pow(2, 33)) / (4 * pointsperleaf * pointsperleaf * 2);
  
  if (size_batch_leaves > leaves) size_batch_leaves = leaves; 
  
  int num_batch_leaves = (leaves) / size_batch_leaves; 
   


	int M_I = M/leaves;

  float del_t1;
  cudaEvent_t t0; 
  cudaEvent_t t1;
   
  int m = min(M_I, 1024);
  //m = min(m, max_nnz);
  int num_blocks = (M_I + 1023) / 1024;
  
  dim3 dimBlock(m, 1, 1);	
  dim3 dimGrid(num_blocks, M_I, size_batch_leaves);
 
  dim3 dimBlock_n(M_I, 1);
  dim3 dimGrid_n(M_I, size_batch_leaves);

  dim3 dimBlock_norm(M_I);
  dim3 dimGrid_norm(size_batch_leaves);






  float *d_K;
  int *d_K_Id;
  checkCudaErrors(cudaMalloc((void **) &d_K, sizeof(float) * pointsperleaf * pointsperleaf * size_batch_leaves));
  checkCudaErrors(cudaMalloc((void **) &d_K_Id, sizeof(int) * pointsperleaf * pointsperleaf * size_batch_leaves));

  checkCudaErrors(cudaEventCreate(&t0));
  checkCudaErrors(cudaEventCreate(&t1));

  checkCudaErrors(cudaEventRecord(t0, 0));
  checkCudaErrors(cudaProfilerStart());
  printf("# leaves : %d \n", leaves);
  printf("# points/leaf : %d \n", pointsperleaf);
  printf(" max_nnz : %d \n", max_nnz);
  printf(" blockDim (norms) : (%d) \n", M_I);
  printf(" blockGrid (norms) : (%d) \n", size_batch_leaves); 
  printf(" blockDim (distance) : (%d,%d,1) \n", m, 1);
  printf(" blockGrid (distance) : (%d,%d,%d) \n", num_blocks, M_I, size_batch_leaves);
  printf(" blockDim (find knn) : (%d,%d,1) \n", M_I, 1);
  printf(" blockGrid (find knn) : (%d,%d,1) \n", M_I, size_batch_leaves);
  printf(" num leaves per loop : %d \n",size_batch_leaves);
  printf(" # points = %d \n" , M);  
  

  for (int leaf_id_g = 0; leaf_id_g < num_batch_leaves; leaf_id_g++){
    compute_norm <<< dimGrid_norm, dimBlock_norm >>>(R, C, V, G_Id, d_K, M_I, leaf_id_g, k);
    checkCudaErrors(cudaDeviceSynchronize()); 
    compute_dist <<< dimGrid, dimBlock >>>(R, C, V, G_Id, d_K, d_K_Id, m, k, M_I, leaf_id_g, max_nnz, M);
    checkCudaErrors(cudaDeviceSynchronize());
    //find_neighbor <<< dimGrid_n, dimBlock_n >>>(knn, knn_Id, d_K, d_K_Id, G_Id, k, M_I, leaf_id_g, M);
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
    int M = 1024*2048;     // total number of points 
    int leaves = 2048;     // number of leaves
    d = 10000;
    int k = 32;
    nnzperrow = 16;
    int max_nnz = 2*nnzperrow;
    
    

    int *d_knn_Id;
    float *d_knn;

    h_R = (int *)malloc(sizeof(int)*(M+1));
    h_G_Id = (int *)malloc(sizeof(int)*(M));

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
