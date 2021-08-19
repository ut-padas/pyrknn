
#include <stdio.h> 
#include <stdlib.h>
//#include <cusparse.h>
#include <cuda.h>
#include <helper_cuda.h>
#include <cuda_runtime.h>
#include <algorithm>



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


__global__ void compute_dist(int* R, int* C, float* V, int* G_Id,  float* K, int m_i, int m_j , int k_nn, int M_I, int leaf_batch_g, int max_nnz, int M){


    // triangular partitioning 

    int b_i = blockIdx.y;
    int b_j = blockIdx.x;
  
    int b_ind = b_i * gridDim.x + b_j;  
    int num_blocks = M_I / m_i; 
  
    float tmp = num_blocks * num_blocks - b_ind - 1;
    int blockId_I = sqrt(tmp);
    blockId_I = num_blocks - 1 - blockId_I;
    int blockId_J_tmp = b_ind - num_blocks*num_blocks + (num_blocks - blockId_I)*(num_blocks - blockId_I) + 2*blockId_I;
    int blockId_J = (blockId_J_tmp+1)/2;
    //printf("(%d,%d) -> %d -> (%d,%d, %d) \n", b_i,b_j,b_ind, blockId_I, blockId_J_tmp, blockId_J);  
 
    int i = threadIdx.y;
    int j = threadIdx.x; 
    
    int ind = i * (m_i + 1) + j; 
     
    tmp = -8*ind + 4*m_i*(m_i + 1) - 7;
    int row_Id = sqrt(tmp)/2.0 - 0.5;
    row_Id = m_i - 1 - row_Id;
    int col_Id = ind + row_Id - m_i*(m_i+1)/2 + (m_i - row_Id) * ((m_i - row_Id) + 1)/2; 
     
    int tmp1;
    bool lower_block = false;
    if (blockId_I % 2 == 0 && blockId_J_tmp % 2 != 0){
      tmp1 = row_Id;
      row_Id = col_Id; 
      col_Id = tmp1;
      lower_block = true;
      //printf("comp1 : (%d,%d), true=(%d,%d) \n", blockId_I, blockId_J, blockId_I, blockId_J_tmp);
    }
    if (blockId_I % 2 != 0 && blockId_J_tmp % 2 != 0){
      tmp1 = row_Id; 
      row_Id = col_Id;
      col_Id = tmp1;
      lower_block = true;
      //printf("comp2 : (%d,%d), true=(%d,%d) \n", blockId_I, blockId_J, blockId_I, blockId_J_tmp);
    }
    //printf("(%d, %d), lower %d \n", blockId_I, blockId_J_tmp, lower_block);
    //if (blockId_I == 1 && blockId_J == 1 && lower_block == 0) printf("(%d,%d) -> %d -> (%d,%d) , lower %d\n", i,j,ind, row_Id,col_Id, lower_block);  
     
    // end of partioning    
    int leaf_id_g = leaf_batch_g * gridDim.z + blockIdx.z;
     
    int g_rowId_I = leaf_id_g * M_I + blockId_I * m_i + row_Id;
    int g_rowId_J = leaf_id_g * M_I + blockId_J * m_j + col_Id;
    //printf(" leaf = %d \n", leaf_id_g);
    //if (leaf_id_g == 2047) printf("(%d,%d) \n", g_rowId_I, g_rowId_J);
    if (g_rowId_I >= M || g_rowId_J >= M) return;
    //printf("m_i = %d , m_j = %d , (%d,%d), blockId_I = %d, blockId_J = %d , g_rowId_I = %d , g_rowId_J = %d \n", m_i,m_j,row_Id,col_Id,blockId_I,blockId_J,g_rowId_I,g_rowId_J);
    //if (g_rowId_I < 110 && g_rowId_I > 90 && g_rowId_J < 110 && g_rowId_J > 90) printf("g_rowId_I = %d , g_rowId_J = %d \n", g_rowId_I, g_rowId_J); 
    int g_Id_i = G_Id[g_rowId_I]; 
    int g_Id_j = G_Id[g_rowId_J];     

    //int g_Id_i = g_rowId_I;
    //int g_Id_j = g_rowId_J;
    
    int ind0_i = R[g_Id_i];
    int ind1_i = R[g_Id_i + 1];

    int ind0_j = R[g_Id_j];
    int ind1_j = R[g_Id_j + 1];
 
    int nnz_i = ind1_i - ind0_i;
    int nnz_j = ind1_j - ind0_j;


    float norm_ij = 0.0;    

    __shared__ int si[8192];
    //__shared__ int sj[4096];

    
    int shift_i = max_nnz * row_Id;
    //int shift_j = max_nnz * col_Id;

    int start_i, start_j; 

    start_i = (lower_block) ? col_Id : col_Id-row_Id;
    //start_j = (lower_block) ? row_Id-col_Id : row_Id;

    //if (leaf_id_g == 0) printf("Id_y = %d , shift = %d \n", threadIdx.y , shift);
    //for (int n_i = 0; n_i < nnz_i; n_i++) norm_ij += V[ind0_i + n_i] * V[ind0_i + n_i];


    //for (int n_i = 0; n_i < nnz_i; n_i++) norm_ij += V[ind0_i + n_i] * V[ind0_i + n_i];
    int ind_read_norm_I = blockIdx.z * M_I + blockId_I * m_i + row_Id;
    int ind_read_norm_J = blockIdx.z * M_I + blockId_J * m_j + col_Id;
    norm_ij += K[ind_read_norm_I];
    norm_ij += K[ind_read_norm_J];

    int blockdim_comp_x = (lower_block) ? row_Id+1 : m_i - row_Id; 
    //int blockdim_comp_y = (lower_block) ? m_j - col_Id : col_Id+1;


    //if (g_rowId_I == 1 && g_rowId_J == 1) printf("lower = %d, c_x = %d , c_y = %d \n", lower_block, blockdim_comp_x, blockdim_comp_y); 
    //for (int n_i = threadIdx.x; n_i < nnz_i; n_i += blockDim.x) si[shift_i + n_i] = C[ind0_i + n_i];
    //for (int n_i = col_Id; n_i < nnz_i; n_i += blockdim_comp_x) si[shift_i + n_i] = C[ind0_i + n_i];

    
    //for (int n_i = start_i; n_i < nnz_i; n_i += blockdim_comp_x) si[shift_i + n_i] = C[ind0_i + n_i]; 
    int ind_read = (b_j < m_i) ? b_i : blockDim.y;
    int shift_read = (b_j % 2 == 0) ? blockDim.y+1 : blockDim.y;
    //for (int n_i = ind_read; n_i < nnz_i; n_i += blockDim.y) si[shift_i + n_i] = C[ind0_i + n_i]; 
    for (int n_i = ind_read; n_i < nnz_i; n_i += shift_read) si[shift_i + n_i] = C[ind0_i + n_i]; 
    /*
    for (int n_i = start_i; n_i < nnz_i; n_i += blockdim_comp_x) {
       si[shift_i + n_i] = C[ind0_i + n_i]; 
       //if (g_rowId_I == 1 && g_rowId_J == 1) printf("row=%d, col=%d, n_i = %d, c=%d, v = %.4f, block_i = %d, block_j = %d \n", row_Id, col_Id, n_i, si[shift_i + n_i],V[ind0_i + n_i], blockId_I, blockId_J);
    }
    */
    //for (int n_j = 0; n_j < nnz_j; n_j++) norm_ij += V[ind0_j + n_j] * V[ind0_j + n_j];
    //__syncthreads();
    //for (int n_j = start_j; n_j < nnz_j; n_j += blockdim_comp_y) sj[shift_j + n_j] = C[ind0_j + n_j];
    /*
    for (int n_j = start_j; n_j < nnz_j; n_j += blockdim_comp_y) {
      sj[shift_j + n_j] = C[ind0_j + n_j];
      if (g_rowId_J == 1) printf("writing sm j, (%d,%d), start_j = %d, nnz_j=%d, move=%d, shift_j = %d, n_j=%d \n", row_Id,col_Id,start_j,nnz_j,blockdim_comp_y, shift_j,n_j);  
      
     //if (g_rowId_I == 1 && g_rowId_J == 1) printf("row=%d, col=%d, n_j = %d, c=%d, v = %.4f\n", row_Id, col_Id, n_j, sj[shift_j + n_j],V[ind0_j + n_j]); 
    }
    */
    //si[n_i + shift_i] = C[ind0_i + n_i];
    //for (int n_j = 0; n_j < nnz_j; n_j++){
    //  sj[n_j + shift_j] = C[ind0_j + n_j];
    //  norm_ij += V[ind0_j + n_j] * V[ind0_j + n_j];
    //}
    __syncthreads();
    //int ValPad_i = si[nnz_i + shift_i -1];
    //int ValPad_j = sj[nnz_j + shift_j -1];

    //for (int n_i = nnz_i ; n_i < max_nnz; n_i++) si[n_i + shift_i] = d;
    //for (int n_j = nnz_j ; n_j < max_nnz; n_j++) sj[n_j + shift_j] = d;
    float c_tmp = 0.0;
    
    int tmp_0, tmp_1, ind_jk, k, ret, testInd;
    
    ret=0; 
    testInd = 0;

    
    for (int pos_k=0; pos_k<nnz_j;pos_k++){       
    //for (int pos_k=0; pos_k<max_nnz;pos_k++){       
        
        k = C[ind0_j + pos_k];
        //k = sj[pos_k + shift_j];
            
        // Binary search 
        for (int l=nnz_i-ret; l > 1; l/=2){
            tmp_0 = ret+l;
            tmp_1 = nnz_i-1;
            testInd = (tmp_0 < tmp_1) ? tmp_0: tmp_1;
            ret = (si[testInd+ shift_i] <= k) ? testInd : ret ;
        }
        tmp_0 = ret+1;
        tmp_1 = nnz_i-1;
        testInd = (tmp_0 < tmp_1) ? tmp_0: tmp_1;
        ret = (si[testInd + shift_i] <= k) ? testInd : ret;
        ind_jk = (si[ret + shift_i] == k) ? ret : -1;
        c_tmp += (ind_jk != -1) ? V[ind0_j + pos_k]*V[ind0_i + ind_jk] : 0;
        
    }
    c_tmp = -2*c_tmp + norm_ij;
    c_tmp = (c_tmp > 0) ? sqrt(c_tmp) : 0.0;
    
	  int col_write = blockId_J * m_j + col_Id; 
	  int row_write = blockId_I * m_i + row_Id;
	  int ind_write = blockIdx.z * M_I * M_I + row_write * M_I + col_write;
	  int col_write_T = blockId_I * m_i + row_Id; 
	  int row_write_T = blockId_J * m_j + col_Id;
	  int ind_write_T = blockIdx.z * M_I * M_I + row_write_T * M_I + col_write_T;
    //printf("thread (%d, %d) -> (%d) -> (%d, %d), block (%d, %d) -> (%d) -> (%d,%d)\n", i,j,ind,row_Id,col_Id, b_i,b_j,b_ind,blockId_I,blockId_J);
    if (lower_block == 1 && row_Id != col_Id && row_Id != m_i && col_Id != 0) K[ind_write] = c_tmp;
    if (lower_block == 0) K[ind_write] = c_tmp;
    if (lower_block == 0 && g_rowId_I != g_rowId_J) K[ind_write_T] = c_tmp;
    if (lower_block == 1 && row_Id != col_Id && row_Id != 0 && col_Id != 0) K[ind_write_T] = c_tmp;
    //printf("row_write = %d,  col_write = %d \n", row_write, col_write);
    //if (g_rowId_I == 32 && g_rowId_J == 32) printf("(%d, %d) = %.4f \n", row_write, col_write, c_tmp); 
    //if (g_rowId_I == 32 && g_rowId_J  == 33) printf("(%d, %d) = %.4f \n", row_write, col_write, c_tmp); 
    //if (row_write_T == 33 && col_write_T == 32) printf("(%d, %d) = %.4f \n", row_write_T, col_write_T, c_tmp);
    /*
    // bitonic sort 
    __shared__ float kvals[4096];
    //__shared__ int id_k[2048];

    int ind_s = threadIdx.y * blockDim.x + threadIdx.x; 
    kvals[ind_s] = c_tmp;
    //id_k[ind_s] = g_rowId_J;
    si[ind_s] = col_write;
    
    
    __syncthreads();
     
    int log_size = 0;
    int m_j_tmp = m_j;
    while (m_j_tmp >>= 1) ++log_size;

    //int log_size = log2(m_j);
    int size = (pow(2,log_size) < m_j) ? pow(2, log_size+1) : m_j;
    // bitonic sort  
    float tmp_f;
    int tmp_i, i;
    i = col_Id;
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
	    //int ind_write = leaf_id_g * M_I * k_nn + row_write * k_nn + col_write; 
	    int ind_write = row_write * k_nn + col_write; 
      //printf("leaf_id = %d , row_write = %d , col_write = %d , ind_write = %d \n", leaf_id_g, row_write, col_write , ind_write);
	    K[ind_write] = kvals[ind_s];
	    //K_ID[ind_write] = id_k[ind_s];
	    //K_ID[ind_write] = si[ind_s];
    }
    */ 
    
}


//__global__ void find_neighbor(float* knn, int* knn_Id, float* K, int* K_Id, int k, int M_I){
__global__ void find_neighbor(float* knn, int* knn_Id, float* K, int* G_Id, int k, int M_I, int m_j, int leaf_batch_g, int M){

    int col_Id = threadIdx.x; 
    int row_Id = blockIdx.x;

    if (row_Id >= M || col_Id >= M) return;
 
    __shared__ float Dist[2048];
    __shared__ int Dist_Id[2048];

    int size = blockDim.x;
    int leaf_id_g = leaf_batch_g * gridDim.y + blockIdx.y; 
    int ind_K = blockIdx.z * M_I * M_I + row_Id * M_I + col_Id; 
    int i = col_Id;
    //if (leaf_id_g == 2047) printf("row = %d , col = %d 0 , ind_shared = %d \n", row_write, col_write, ind_shared);
    //Dist[ind_shared] = (col_Id < k) ? knn[ind_knn] : (col_Id < size) ? K[ind_read] : 1e30;
    Dist[col_Id] = K[ind_K];
    //Dist_Id[ind_shared] = (col_Id < k) ? knn_Id[ind_knn] : (col_Id < size) ? K_Id[ind_read] : 0;
    //Dist_Id[ind_shared] = (col_Id < k) ? 0 : (col_Id < size) ? K_Id[ind_read] : 0;
    //Dist_Id[col_Id] = col_Id;
    Dist_Id[col_Id] = G_Id[leaf_id_g * M_I + col_Id];
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

    /*
    if (col_Id < k) {
      K[ind_K] = Dist[col_Id];
      K_Id[ind_K] = Dist_Id[col_Id];
    }
    */

    size = 2*k;


    int ind_knn = leaf_id_g * M_I * k + row_Id * k + col_Id;

    if (col_Id >= k && col_Id < size) Dist[col_Id] = 1e30;
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

  __shared__ float Dist[4096];
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
  std::random_shuffle(&G_Id[0], &G_Id[M]);
  /*
  for (int m = 0; m < M; m++){ 
  printf("G_Id[%d] = %d \n", m , G_Id[m]);
  } 
  */
}

void gpu_knn(int *R, int *C, float *V, int *G_Id, int M, int leaves, int k, float *knn, int *knn_Id, int max_nnz, int d){
 
	int pointsperleaf = M/leaves;
	int m_i = 8192 / max_nnz;
  m_i = min(m_i, pointsperleaf);
  int m_j = m_i;
  //m_j = min(m_j, pointsperleaf);
  if (m_i > 32){ 
    //m_j = 1024/m_i; 
    m_i = 32; 
    m_j = 32;
  } 
  if (m_j * max_nnz > 4096 || m_i * max_nnz > 8192) printf("Exceeds the shared memory size \n"); 
	int num_batch_I = (pointsperleaf + m_i - 1) / m_i;
	int num_batch_J = (pointsperleaf + m_j - 1) / m_j;

  int size_batch_leaves = (pow(2, 33)) / ( 4* pointsperleaf * pointsperleaf) ;
  int num_batch_leaves = (leaves + size_batch_leaves - 1) / size_batch_leaves;
	int M_I = M/leaves;

  //printf("%d , %d  , %d \n", num_batch_I, num_batch_J, num_batch_leaves);
  float del_t1;
  cudaEvent_t t0; 
  cudaEvent_t t1;
  int block_size_i = m_i / 2;
  int block_size_j = m_i + 1; 
  
  dim3 dimBlock(block_size_j, block_size_i, 1);	
  //dim3 dimBlock(32, 32, 1);	
  dim3 dimGrid(num_batch_J, num_batch_I, size_batch_leaves); 
  dim3 dimBlock_n(M_I, 1);
  dim3 dimGrid_n(M_I, size_batch_leaves);

  dim3 dimBlock_merge(2*k);
  dim3 dimGrid_merge(M_I);
  dim3 dimBlock_norm(M_I);
  dim3 dimGrid_norm(size_batch_leaves);


  float *d_K;
  checkCudaErrors(cudaMalloc((void **) &d_K, sizeof(float) * pointsperleaf * pointsperleaf * size_batch_leaves));


  checkCudaErrors(cudaEventCreate(&t0));
  checkCudaErrors(cudaEventCreate(&t1));

  checkCudaErrors(cudaEventRecord(t0, 0));
  for (int leaf_id_g = 0; leaf_id_g < num_batch_leaves; leaf_id_g++){
    printf("running leaf=%d \n", leaf_id_g);
    compute_norm <<< dimGrid_norm, dimBlock_norm >>>(R, C, V, G_Id, d_K, M_I, leaf_id_g);
    checkCudaErrors(cudaDeviceSynchronize());
    compute_dist <<< dimGrid, dimBlock >>>(R, C, V, G_Id, d_K, m_i, m_j, k, M_I, leaf_id_g, max_nnz, M);
    checkCudaErrors(cudaDeviceSynchronize());
    find_neighbor <<< dimGrid_n, dimBlock_n >>>(knn, knn_Id, d_K, G_Id, k, M_I, m_j, leaf_id_g, M);
  } 
  
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaEventRecord(t1, 0));
  checkCudaErrors(cudaEventSynchronize(t1));
  checkCudaErrors(cudaEventElapsedTime(&del_t1, t0, t1));
  printf("# leaves : %d \n", leaves);
  printf("# points/leaf : %d \n", pointsperleaf);
  printf("  max_nnz : %d \n", max_nnz);
	printf("blockDim (distance) : (%d,%d,1) \n", block_size_j, block_size_i);
  printf("blockGrid (distance) : (%d,%d,%d) \n", num_batch_J, num_batch_I, size_batch_leaves);
	printf("blockDim (find knn) : (%d, 1) \n", M_I);
  printf("blockGrid (find knn) : (%d,%d) \n", M_I, size_batch_leaves);
  printf("\n Elapsed time (s) : %.4f \n ", del_t1/1000);
  printf(" # points = %d" , M);
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
    int M = 1024*2048;
    int leaves = 2048;
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
    for (int i = 32; i < 34; i++){
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

    printf("Random csr is generated  \n");

    gpu_knn(d_R, d_C, d_V, d_G_Id, M, leaves, k, d_knn, d_knn_Id, max_nnz, d);
    
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
