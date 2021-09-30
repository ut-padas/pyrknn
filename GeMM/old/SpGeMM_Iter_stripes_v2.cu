
#include <stdio.h> 
#include <stdlib.h>
//#include <cusparse.h>
#include <cuda.h>
#include <helper_cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include "knn_seq.cpp"
#include <limits.h>
#include <curand.h>
#include <curand_kernel.h>
//#include <math.h>

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

}


__global__ void knn_kernel_tri(int* R, int* C, float* V, int* G_Id, float* Norms , int k_nn, float* KNN_dist, int* KNN_Id, int ppl, int max_nnz, int m, int* sort_arr, int* sort_arr_part, int steps) {

  __shared__ int SM[4096];
  __shared__ float SM_dist[2048];
  __shared__ int SM_Id[2048];



  int ind = threadIdx.x;
  int leaf_id_g = blockIdx.y;  
  int block = blockIdx.x;

  
  int i = ind / k_nn;
  int j = ind - i * k_nn;
  
  int g_rowId = leaf_id_g * ppl + block * k_nn + i;
  int g_colId = leaf_id_g * ppl + block * k_nn + j;

  int perm_i = G_Id[g_rowId];
  int perm_j = G_Id[g_colId];

  int ind0_i = R[perm_i];
  int ind1_i = R[perm_i + 1];

  int ind0_j = R[perm_j];
  int ind1_j = R[perm_j + 1];
  
  int nnz_i = ind1_i - ind0_i;
  int nnz_j = ind1_j - ind0_j;
  
  
  float norm_ij = 0.0;
  
  
  norm_ij = Norms[g_rowId] + Norms[g_colId];

  int shift_i = max_nnz * i;
  

  for (int n_i = j; n_i < nnz_i; n_i += k_nn) SM[shift_i + n_i] = C[ind0_i + n_i];
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
    SM_dist[i *  k_nn + j] = c_tmp;
    SM_Id[i *  k_nn + j] = perm_j;
    if (j > i) SM_dist[j  * k_nn + i] = c_tmp;
    if (j > i) SM_Id[j *  k_nn + i] = perm_i;
  }

  __syncthreads();

  
  /*
  if (j >= i) SM_dist[i * 2 * m + j] = c_tmp;
  if (j >= i) SM[i * 2 * m + j] = G_Id[leaf_id_g * ppl + block * m + j];
  
  if (j > i) SM_dist[j * 2 * m + i] = c_tmp;
  if (j > i) SM[j * 2 * m + i] = G_Id[leaf_id_g * ppl + block * m + i];
  */
  
  //if (block * k_nn + i == 63 && i == j) printf("(%d , %d ) , %.4f ,||= %.4f \n" , i , j , c_tmp, norm_ij);

  
  
  
  


           
      // TODO : should fix the issue for the initial value
      //int ind_knn = leaf_id_g * ppl * k_nn + (block * m + row_read) * k_nn + col_read;
      //SM_dist[row_read * 2 * m + col_read + m] = (col_read < k_nn) ? KNN_dist[ind_knn] : 1e30;
      //SM[row_read * 2 * m + col_read + m] = (col_read < k_nn) ? KNN_Id[ind_knn] :  0;
      
      //SM_dist[i * 2 * m + j + m] = (j < k_nn) ? 1e30 : 1e30;
      //SM[i * 2 * m + j + m] = (j < k_nn) ? 0 :  0;
     
       

 
      //__syncthreads();
      
      // start efficient sort for 2m length array

      //int size = 2 * m;
      //int len = (size/2) * steps;
      //for (int s_ind = ind; s_ind < len; s_ind += blockDim.x) {
        //if (ind < len) SM_sort_arr[ind] = sort_arr[ind];
        //if (ind < len) SM_sort_arr_part[ind] = sort_arr_part[ind];
         
      //}
      //for (int s_ind = ind; s_ind < len; s_ind += blockDim.x) SM_sort_arr_part[s_ind] = sort_arr_part[s_ind];
      
      //__syncthreads();
      /*
      float tmp_f;
      int tmp_i;
      for (int step = 0; step < steps; step ++){
       
        int j_sort = SM_sort_arr[step * m + j];
        int ixj_sort = SM_sort_arr_part[step * m + j];
        int min_max = (1 & j_sort);
        j_sort = j_sort >> 1;
        ixj_sort = ixj_sort >> 1;
        
        int j_sort_tmp = i * 2 * m + j_sort;
        int ixj_sort_tmp = i * 2 * m + ixj_sort;      
        //if (block * m + i == 63) printf("%d - %d , minmax = %d \n", j_sort, ixj_sort, min_max); 
        if (min_max) {
          if (SM_dist[j_sort_tmp] > SM_dist[ixj_sort_tmp]){
            
            tmp_f = SM_dist[ixj_sort_tmp];
            SM_dist[ixj_sort_tmp] = SM_dist[j_sort_tmp];
            SM_dist[j_sort_tmp] = tmp_f;

            tmp_i = SM[ixj_sort_tmp];
            SM[ixj_sort_tmp] = SM[j_sort_tmp];
            SM[j_sort_tmp] = tmp_i;
         
          }
        } else {
          if (SM_dist[j_sort_tmp] < SM_dist[ixj_sort_tmp]){
            
            tmp_f = SM_dist[ixj_sort_tmp];
            SM_dist[ixj_sort_tmp] = SM_dist[j_sort_tmp];
            SM_dist[j_sort_tmp] = tmp_f;

            tmp_i = SM[ixj_sort_tmp];
            SM[ixj_sort_tmp] = SM[j_sort_tmp];
            SM[j_sort_tmp] = tmp_i;

          }
        }
      __syncthreads();
      }

      */
      // end of effificent bitonic sort

      //if (block * m + i == 63) printf("end  %.4f at %d , %d \n", SM_dist[i * 2 * m + j], SM[i * 2 * m + j], j);


      /*
      // bitonic sort 

            

      //for (int batch = 0; batch < 2; batch ++){
       
      //int row = ind / (2 * m);
      //int col = ind - row * 2 * m;



      //if (leaf_id_g == 1000 && w == 1000 ) printf("val = %.4f , id = %d \n", SM_dist[col], SM[col]);
      float tmp_f; 
      int tmp_i;
      int size = 2 *m;
      //int col_tmp = (row + batch * m / 2) * 2 * m + col;
      //int col_tmp = (row) * 2 * m + col;
      int col_tmp = (i) * 2 * m + j;
      for (int g = 2; g <= size; g *= 2){
        for (int l = g/2; l > 0; l /= 2){

          int ixj = j ^ l;
          int ixj_tmp = (i) * 2 * m + ixj;

          if (ixj > j){
            if(( j & g) == 0){
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

   */    
   
   
   
   
   if (j < k_nn){
    int ind_knn = leaf_id_g * ppl * k_nn + (block * k_nn + i) * k_nn + j;  
    KNN_dist[ind_knn] = SM_dist[ i * k_nn + j];
    KNN_Id[ind_knn] = SM_Id[i  * k_nn + j]; 
    
    
   }

}




__global__ void knn_kernel_sq(int* R, int* C, float* V, int* G_Id, float* Norms, int k_nn, float* KNN_dist, int* KNN_Id, int ppl, int max_nnz, int m, int blockInd, int* sort_arr, int* sort_arr_part, int steps, float* d_knn_temp) {


  __shared__ int SM[1024];
  __shared__ float SM_dist[4096];
  __shared__ int SM_Id[4096];
  
  int row_l = blockIdx.x;
  int leaf_id_g = blockIdx.y;
  int j = threadIdx.x;
  //int size_sort = 2 * blockDim.x;
  
  //int size_sort_step = size_sort - k_nn; 
  int size_part = ppl - (k_nn) * blockInd; 

  int rowId_leaf = k_nn * blockInd + row_l;
  int g_rowId_I = leaf_id_g * ppl + rowId_leaf;
  int perm_i = G_Id[g_rowId_I];
  int ind0_i = R[perm_i];
  int ind1_i = R[perm_i+1];
  float norm_i = Norms[g_rowId_I];
  int nnz_i = ind1_i - ind0_i;


  for (int n_i = j; n_i< nnz_i; n_i += blockDim.x) SM[n_i] = C[ind0_i + n_i];
   
  //int num_batches = size_part / (size_sort);
  __syncthreads();

  //if (leaf_id_g == 1000 && rowId_leaf == 1000 && j == 0) printf("size_part = %d , size_sort = %d , num_batches = %d \n", size_part, size_sort, num_batches);
  for (int pt = j; pt < size_part; pt += blockDim.x){


    int colId_leaf = k_nn * (blockInd) + pt;
      
      
      
    if (colId_leaf >= k_nn * (blockInd+1) && colId_leaf < ppl){
      
      int g_rowId_J = leaf_id_g * ppl + colId_leaf;
  
      int perm_j = G_Id[g_rowId_J];
      int ind0_j = R[perm_j];
      int ind1_j = R[perm_j+1];

      int nnz_j = ind1_j - ind0_j;

      float norm_ij = norm_i + Norms[g_rowId_J];
      
      float c_tmp = 0.0;
      int tmp_0, tmp_1, ind_jk, k, ret, testInd;
    
      ret = 0;
      testInd = 0;
      // loop over the elements of j
      
      for (int pos_k = 0; pos_k < nnz_j; pos_k++){
        
        k = C[ind0_j + pos_k];
    
        // Binary search
    
        for (int l = nnz_i - ret; l > 1; l /= 2){

          tmp_0 = ret + l;
          tmp_1 = nnz_i - 1;
          testInd = (tmp_0 < tmp_1) ? tmp_0 : tmp_1;
          //ret = (SM[testInd + shift_i] <= k) ? testInd : ret;
          ret = (SM[testInd] <= k) ? testInd : ret;
        
        }

        tmp_0 = ret + 1;
        tmp_1 = nnz_i - 1;
        testInd = (tmp_0 < tmp_1 ) ? tmp_0 : tmp_1;
        //ret = (SM[testInd + shift_i] <= k) ? testInd : ret;
        //ind_jk = (SM[ret + shift_i] == k) ? ret : -1;
        ret = (SM[testInd] <= k) ? testInd : ret;
        ind_jk = (SM[ret] == k) ? ret : -1;
        c_tmp += (ind_jk != -1) ? V[ind0_j + pos_k] * V[ind0_i + ind_jk] : 0;

      }
    
      c_tmp = -2 * c_tmp + norm_ij;
      c_tmp = ( c_tmp > 0) ? sqrt(c_tmp) : 0.0;

      SM_dist[pt] = c_tmp;
      SM_Id[pt] = perm_j;
      
      int size_tmp = size_part - k_nn;
      int ind_tmp = leaf_id_g * k_nn * size_tmp + row_l * size_tmp + colId_leaf - (k_nn) * (blockInd+1);
      
      d_knn_temp[ind_tmp] = c_tmp; //SM_dist[j_tmp];
      //d_knnId_temp[ind_tmp] = perm_i;
      
      
      //__syncthreads();
    } else {
        
      SM_dist[pt] = (pt < k_nn) ? KNN_dist[leaf_id_g * ppl * k_nn + rowId_leaf * k_nn + pt] : 1e30;
      SM_Id[pt] = (pt < k_nn) ? KNN_Id[leaf_id_g * ppl * k_nn + rowId_leaf * k_nn + pt] : 0;
      
        
    }
    
      
      
      
      
  }
    
    
    

  float tmp_f;
  int tmp_i;
  int ind_sort;
    
  __syncthreads();
    
  for (int step = 0 ; step < steps; step++){
    for (int pt = j; pt < size_part; pt += blockDim.x){
      ind_sort = step * size_part + pt;
    
      int tid = sort_arr[ind_sort];
      int ixj = sort_arr_part[ind_sort];
    
      int min_max = (1 & tid);
      int coupled_flag = (1 & ixj);
      //if (leaf_id_g == 1000 && rowId_leaf == 1000 && j == blockDim.x-1) printf(" step %d , %d - %d , min_max %d , c %d \n", step, tid, ixj, min_max, coupled_flag);
      tid = tid >> 1;
      ixj = ixj >> 1;
    
      if (coupled_flag == 1){
        
        ind_sort += blockDim.x;
        //int tid_1 = sort_arr[ind_sort];
        //int ixj_1 = sort_arr_part[ind_sort];
        int tid_1 = sort_arr[ step * 2 * blockDim.x + j + blockDim.x];
        int ixj_1 = sort_arr_part[step * 2 * blockDim.x + j + blockDim.x];
        int min_max_1 = (1 & tid_1);
        
        
        tid_1 = tid_1 >> 1;
        ixj_1 = ixj_1 >> 1;
        

        if (min_max_1 == 1 && SM_dist[tid_1] > SM_dist[ixj_1]){
        
          tmp_f = SM_dist[tid_1];
          SM_dist[tid_1] = SM_dist[ixj_1];
          SM_dist[ixj_1] = tmp_f;
        
          tmp_i = SM_Id[tid_1];
          SM_Id[tid_1] = SM_Id[ixj_1];
          SM_Id[ixj_1] = tmp_i;
                  
        }

        if (min_max_1 == 0 && SM_dist[tid] < SM_dist[ixj]){
        
          tmp_f = SM_dist[tid_1];
          SM_dist[tid_1] = SM_dist[ixj_1];
          SM_dist[ixj_1] = tmp_f;
        
          tmp_i = SM_Id[tid_1];
          SM_Id[tid_1] = SM_Id[ixj_1];
          SM_Id[ixj_1] = tmp_i;
                  
        }
        
      } 
        
        //if (leaf_id_g == 1000 && rowId_leaf == 1000 && step == 0) printf("step = %d , %d - %d , min %d , c %d \n", step, tid, ixj, min_max, coupled_flag);
      if (min_max == 1){
        if (SM_dist[tid] > SM_dist[ixj]){
          tmp_f = SM_dist[tid];
          SM_dist[tid] = SM_dist[ixj];
          SM_dist[ixj] = tmp_f;
        
          tmp_i = SM_Id[tid];
          SM_Id[tid] = SM_Id[ixj];
          SM_Id[ixj] = tmp_i;
        } 
      } else {
        if (SM_dist[tid] < SM_dist[ixj]){
          tmp_f = SM_dist[tid];
          SM_dist[tid] = SM_dist[ixj];
          SM_dist[ixj] = tmp_f;
        
          tmp_i = SM_Id[tid];
          SM_Id[tid] = SM_Id[ixj];
          SM_Id[ixj] = tmp_i;
        }
      }
    }
    __syncthreads();
  
  }
    
    
  /*
  if (j < k_nn){
    
    res_dist[col_batch * k_nn + j] = SM_dist[j];
    res_Id[col_batch * k_nn + j] = SM_Id[j];
    //if (leaf_id_g == 1000 && rowId_leaf == 1000) printf(" blockInd = %d , col_batch = %d, %d , %.4f , %d \n", blockInd , col_batch, col_batch * k_nn + j , res_dist[col_batch * k_nn + j], res_Id[col_batch * k_nn + j]);
  }
  */
    
    


  


  
    
  
  if (j < k_nn){
    int ind_knn = leaf_id_g * ppl * k_nn + rowId_leaf * k_nn + j;
    //if (leaf_id_g == 1000 && rowId_leaf == 1000) printf("blockInd = %d , j = %d , %.4f , %d \n", blockInd, j, res_dist[j], res_Id[j]);
    KNN_dist[ind_knn] = SM_dist[j];
    KNN_Id[ind_knn] = SM_Id[j];
    
  }


}

__global__ void knn_kernel_v_reduced(float* KNN, int* KNN_Id, int k_nn, int m, int ppl, int blockInd, float* d_temp_knn,int* d_temp_knnId, int* sort_arr, int* sort_arr_part, int steps, int* G_Id){

  __shared__ float SM_dist[1024];
  __shared__ int SM_Id[1024];


  int j = threadIdx.x;

  int col = blockIdx.x;
  int leaf_id_g = blockIdx.y;
  int colId_leaf = col + k_nn * (blockInd + 1);
  int size_part = ppl - (blockInd + 1) * (k_nn);


  int ind_tmp = leaf_id_g * k_nn * size_part + j * size_part + col;
  SM_dist[j] = d_temp_knn[ind_tmp];
  SM_Id[j] = d_temp_knnId[ind_tmp];
  //SM_Id[j] = G_Id[leaf_id_g * ppl + j +]

  int ind_knn = leaf_id_g * k_nn * ppl + colId_leaf * k_nn + j;
  SM_dist[j + k_nn] = KNN[ind_knn];
  SM_Id[j + k_nn] = KNN_Id[ind_knn];

  __syncthreads();

  float tmp_f;
  int tmp_i;
  //if (leaf_id_g == 1000 && blockInd == 0 && colId_leaf == 1000) printf("%d , %.4f , %d \n", j, SM_dist[j], SM_Id[j]);
  //if (leaf_id_g == 1000 && blockInd == 0 && colId_leaf == 1000) printf("%d , %.4f , %d \n", j+k_nn, SM_dist[j+k_nn], SM_Id[j+k_nn]);

  for (int step = 0 ; step < steps; step++){

    int tid = sort_arr[step * 2 * blockDim.x + j];
    int ixj = sort_arr_part[step * 2 * blockDim.x + j];
    int min_max = (1 & tid);
    int coupled_flag = (1 & ixj);

    tid = tid >> 1;
    ixj = ixj >> 1;
    if (coupled_flag == 1){
        
      
      int tid_1 = sort_arr[step * 2 * blockDim.x + j + blockDim.x];
      int ixj_1 = sort_arr_part[step * 2 * blockDim.x + j + blockDim.x];
      int min_max_1 = (1 & tid_1);
      
      
      
      tid_1 = tid_1 >> 1;
      ixj_1 = ixj_1 >> 1;
      

      if (min_max_1 == 1 && SM_dist[tid_1] > SM_dist[ixj_1]){
      
        tmp_f = SM_dist[tid_1];
        SM_dist[tid_1] = SM_dist[ixj_1];
        SM_dist[ixj_1] = tmp_f;
       
        tmp_i = SM_Id[tid_1];
        SM_Id[tid_1] = SM_Id[ixj_1];
        SM_Id[ixj_1] = tmp_i;
                
      }

      if (min_max_1 == 0 && SM_dist[tid] < SM_dist[ixj]){
      
        tmp_f = SM_dist[tid_1];
        SM_dist[tid_1] = SM_dist[ixj_1];
        SM_dist[ixj_1] = tmp_f;
       
        tmp_i = SM_Id[tid_1];
        SM_Id[tid_1] = SM_Id[ixj_1];
        SM_Id[ixj_1] = tmp_i;
                
      }
      
    } 
    
     
    if (min_max == 1){
      if (SM_dist[tid] > SM_dist[ixj]){
        tmp_f = SM_dist[tid];
        SM_dist[tid] = SM_dist[ixj];
        SM_dist[ixj] = tmp_f;
      
        tmp_i = SM_Id[tid];
        SM_Id[tid] = SM_Id[ixj];
        SM_Id[ixj] = tmp_i;
      } 
    } else {
      if (SM_dist[tid] < SM_dist[ixj]){
        tmp_f = SM_dist[tid];
        SM_dist[tid] = SM_dist[ixj];
        SM_dist[ixj] = tmp_f;
      
        tmp_i = SM_Id[tid];
        SM_Id[tid] = SM_Id[ixj];
        SM_Id[ixj] = tmp_i;
      }
    }



  __syncthreads();

  }
  if (leaf_id_g == 1000 && blockInd == 0 && colId_leaf == 1000) printf("sorted %d , %.4f , %d \n", j, SM_dist[j], SM_Id[j]);
  if (leaf_id_g == 1000 && blockInd == 0 && colId_leaf == 1000) printf("sorted %d , %.4f , %d \n", j+k_nn, SM_dist[j+k_nn], SM_Id[j+k_nn]);

  if (j < k_nn) {
    int ind_knn = leaf_id_g * ppl * k_nn + colId_leaf * k_nn + j;
    KNN[ind_knn] = SM_dist[j];
    KNN_Id[ind_knn] = SM_Id[j];
  
  }

}



//__global__ void knn_kernel_v(float* KNN, int* KNN_Id, int k_nn, int m, int ppl, int blockInd, float* d_temp_knn,int* d_temp_knnId){
__global__ void knn_kernel_v(float* KNN, int* KNN_Id, int k_nn, int m, int ppl, int blockInd, float* d_temp_knn, int* G_Id){

  __shared__ float SM_dist[1024];
  __shared__ int SM_Id[1024];


  int tid = threadIdx.x;

  int col = blockIdx.x;
  int leaf_id_g = blockIdx.y;
  int colId_leaf = col + k_nn * (blockInd + 1);
  int size_part = ppl - (blockInd + 1) * (k_nn);
  
  if (tid < k_nn){
    int ind_tmp = leaf_id_g * k_nn * size_part + tid * size_part + col;
    SM_dist[tid] = d_temp_knn[ind_tmp];
    int rowId_g = leaf_id_g * ppl + k_nn * blockInd + tid;  
    //SM_Id[tid] = d_temp_knnId[ind_tmp];
    SM_Id[tid] = G_Id[rowId_g];
    
  } else {
    int ind_knn = leaf_id_g * k_nn * ppl + colId_leaf * k_nn + tid - k_nn;
    SM_dist[tid] = KNN[ind_knn];
    SM_Id[tid] = KNN_Id[ind_knn];
      
  }

  

  __syncthreads();
  
  

  // sort 
  float tmp_f;
  int tmp_i;

  int size = 2 * k_nn;
  for (int g = 2; g <= size; g *= 2){
    for (int l = g/2; l > 0; l /= 2){

      int ixj = tid ^ l;
      
      if (ixj > tid){
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
      }
      __syncthreads();
      
      
    }
  }
  __syncthreads();
  
  //if (colId_leaf == 63) printf("sorted vertical tid = %d , %.4f , %d \n" , tid , SM_dist[tid], SM_Id[tid]);
  if (tid < k_nn) {
    int ind_knn = leaf_id_g * ppl * k_nn + colId_leaf * k_nn + tid;
    KNN[ind_knn] = SM_dist[tid];
    KNN_Id[ind_knn] = SM_Id[tid];
    
  
  }

}

__global__ void sort_GIds(int* G_Id, int ppl){

  int tid = threadIdx.x;
  
  int leaf_id_g = blockIdx.x;
  

  int elem_Id = leaf_id_g * ppl + tid;
  __shared__ int SM_GId[8192];

  for (int tid_seq = tid; tid_seq < ppl; tid_seq += blockDim.x) SM_GId[tid_seq] = G_Id[leaf_id_g * ppl + tid_seq];
  
  __syncthreads();
  
  int num_batches = ppl / 1024;
  int tmp_i;
  for (int g = 2; g <= ppl; g *= 2){
    for (int l = g/2; l > 0; l /= 2){
      for (int tid_seq = tid; tid_seq < ppl; tid_seq += blockDim.x){
        
        int ixj = tid_seq ^ l;
        //if (leaf_id_g == 0 && g == 2 && l == 1) printf("tid_seq = %d , ixj = %d \n", tid_seq, ixj);
        if (ixj > tid_seq){
          if ((tid_seq & g) == 0){
            if (SM_GId[tid_seq] > SM_GId[ixj]){
              tmp_i = SM_GId[tid_seq];
              SM_GId[tid_seq] = SM_GId[ixj];
              SM_GId[ixj] = tmp_i;
            }
          } else {
            if (SM_GId[tid_seq] < SM_GId[ixj]){
              tmp_i = SM_GId[tid_seq];
              SM_GId[tid_seq] = SM_GId[ixj];
              SM_GId[ixj] = tmp_i;
            }

          }
        }
      }
    __syncthreads();
    }
  }    

  for (int tid_seq = tid; tid_seq < ppl; tid_seq += blockDim.x) {
    if (leaf_id_g == 1000 && tid_seq < 50) printf("tid_seq = %d , Id = %d \n", tid_seq, SM_GId[tid_seq]);
    G_Id[leaf_id_g * ppl + tid_seq] = SM_GId[tid_seq];
  }

}





__global__ void test_sort(int* sort_arr, int* sort_arr_part, int N_true, int N_pow2, int steps, float* list){


  int tid = threadIdx.x;
  /*
  curandState_t state;
  
  
  
  
  
  
  for (int tmp = tid; tmp < N_pow2; tmp++) {
    if (tmp < N_true) *list[tmp] = curand(&state) % RAND_MAX;
    if (tmp >= N_true) list[tmp] = 1e30;
  }
  */

  


  float tmp_f;
  

  for (int step = 0; step < steps ; step++){

    int j = sort_arr[step * N_true + tid];
    int ixj = sort_arr_part[step * N_true + tid];
    int min_max = (1 & j);
    int coupled = (1 & ixj);

    j = j >> 1;
    ixj = ixj >> 1;

    if (coupled == 1){

      int j_1 = sort_arr[step * N_true + tid + blockDim.x];
      int ixj_1 = sort_arr_part[step * N_true + tid + blockDim.x];

      int min_max_1 = (1 & j_1);

      j_1 = j_1 >> 1;
      ixj_1 = ixj_1 >> 1;

      if (min_max_1 == 0 && list[j_1] < list[ixj_1]){

        tmp_f = list[j_1];
        list[j_1] = list[ixj_1];
        list[ixj_1] = tmp_f;

      }
      if (min_max_1 == 1 && list[j_1] > list[ixj_1]){
        tmp_f = list[j_1];
        list[j_1] = list[ixj_1];
        list[ixj_1] = tmp_f;
      }

    }


    if (min_max == 1){
      if (list[j] > list[ixj]){
        
        tmp_f = list[j];
        list[j] = list[ixj];
        list[ixj] = tmp_f;

      }
    } else{
      if (list[j] < list[ixj]){
        
        tmp_f = list[j];
        list[j] = list[ixj];
        list[ixj] = tmp_f;

      }

    }

    


  

    //__syncthreads();  
    //for (int tmp = tid; tmp < N_pow2; tmp += blockDim.x) printf("sorted step = %d/%d , list[%d] = %.4f \n",step, steps, tmp, list[tmp]);
  __syncthreads();



  } 

  for (int tmp = tid; tmp < N_pow2; tmp += blockDim.x) printf("sorted list[%d] = %.4f \n", tmp, list[tmp]);



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



void par_block_indices(int N, int* d_arr){


  int* vals;
  vals = (int *)malloc(sizeof(int) * N * (N-1));

  int elem = -1;
  /*
  do {

     for (int i = 0; i < N; i++)
     {
       if (bitmask[i]) {
       elem++;
       vals[elem] = i;
       }
     }
  } while (std::prev_permutation(bitmask.begin(), bitmask.end()));
  */
  for (int m = 0; m < N; m++){
    for (int n = m+1; n < N; n++){
      elem++;
      vals[elem] = m;
      elem++;
      vals[elem] = n;
    }
  }

  int *arr;
  arr = (int *)malloc(sizeof(int) * N * (N-1));  


  int* track;
  track = (int *) malloc(sizeof(int) * (N) * (N-1));
  memset(track, 0, sizeof(int) * N * (N-1));

  int* row_mems;
  row_mems = (int *) malloc(sizeof(int) * (N-1));
  memset(row_mems, 0, sizeof(int) * (N-1));

  for (int mems = 0; mems < (N/2) * (N-1); mems++){

    int i = vals[2*mems];
    int j = vals[2*mems+1];
    int row = 0;
    //printf("(%d, %d) -> ", i,j);
    bool ex_f = false;
    while (! ex_f) {
      if (track[row * N + i] == 0 && track[row * N + j] == 0){
        int start = row_mems[row];

        arr[row * N + start] = i;
        arr[row * N + start+1] = j;
        track[row * N + i] = 1;
        track[row * N + j] = 1;
        row_mems[row] +=2;
        ex_f = true;
        //printf("at row %d loc = %d \n", row, start);
      }
      row++;
    }
  }
  //printf("N = %d \n", N); 
  checkCudaErrors(cudaMemcpy(d_arr, arr, sizeof(int)*N * (N-1), cudaMemcpyHostToDevice));
  

}
/*
__global__ void precomp_arbsize_sortId_kernel(int* d_arr, int* d_arr_part, int N_true, int N_pow2, int n_s){

  int min_max, elem, coupled_elem;

  int loc_len = N_true /2;
  
  __shared__ int tracker[2048];

  int tid = threadIdx.x;

  for (int i = tid; i < N_pow2; i++) tracker[i] = i;

  int step = 0;

  for (int g = 2; g <= N_pow2; g *= 2){
    for (int l = g/2; l > 0; l /= 2){
      //elem = 0;
      for (int j = 2 * tid; j < 2 * tid + 2; j++){
        int ixj = j ^ l;

        if (tracker[ixj] >= N_true && tracker[j] >= N_true) continue;

        if (ixj > j){
          min_max = ((j&g) == 0 ) ? 1 : 0;

          coupled_elem = 0;
          int write_loc = tid;
          if (tracker[ixj] >= N_true) {
            coupled_elem = 1;
            int tmp = d_arr_part[step * N_true + tid - 1];
            int tmp_f_0 = (1 & tmp);
            int ixj_prev = tmp >> 1;
            if (ixj_prev >= N_true && tmp_f_0 == 1){
              elem -= 1;
              write_loc = tid + loc_len;
            }
          }
          //if (i >= 2048 || ixj >= 2048) printf(" i = %d , ixj = %d , sort_size = %d , N_true = %d , N_pow2 = %d \n", i, ixj, sort_size , N_true , N_pow2);
          d_arr[step * N_true + write_loc] = (j << 1) + min_max;
          d_arr_part[step * N_true + write_loc] = (ixj << 1) + coupled_elem;

        }
      }
     step++;
     __syncthreads();
    }
  }
  


}

*/


void precomp_arbsize_sortId(int* d_arr, int* d_arr_part, int N_true, int N_pow2, int steps, int copy_size){

  
  
  int min_max, elem, coupled_elem;
  int loc_len = ceil(N_true/2);

  int* tracker;
  tracker = (int *)malloc(sizeof(int) * N_pow2);
  
  for (int i = 0; i < N_pow2; i ++) tracker[i] = i;

  int step = 0;

  int *arr, *arr_part;
  arr = (int *)malloc(sizeof(int) * copy_size); 
  arr_part = (int *)malloc(sizeof(int) * copy_size);

  memset(arr, 0, sizeof(int) * copy_size);
  memset(arr_part, 0, sizeof(int) * copy_size);



  int first_pair = 1;
  int prev_elem = 0;
  int tmp2;
  for (int g = 2; g <= N_pow2; g *= 2){
    for (int l = g/2; l > 0; l /= 2){
      elem = 0;
      for (int i = 0; i < N_pow2; i++){
        int ixj = i ^ l;

        if (tracker[ixj] >= N_true && tracker[i] >= N_true) continue;

        if (ixj > i){

          min_max = ((i&g) == 0 ) ? 1 : 0;

          coupled_elem = 0;

          int write_loc = elem;
          
          if (tracker[ixj] >= N_true || tracker[i] >= N_true) {
            coupled_elem = 1;
            if (min_max == 0 && tracker[ixj] >= N_true) {
              tmp2 = tracker[ixj];
              tracker[ixj] = tracker[i];
              tracker[i] = tmp2;
            }
            if (min_max == 1 && tracker[i] >= N_true) {
              tmp2 = tracker[ixj];
              tracker[ixj] = tracker[i];
              tracker[i] = tmp2;
            }
            if (first_pair == 1){
              prev_elem = elem;
              write_loc = elem;
              first_pair = 0;
            } else {
              write_loc = prev_elem + loc_len;
              first_pair = 1;
              elem++;
            }
            //if (ixj > 2048 || i > 2048) printf("err \n");
            //printf("step = %d , elem = %d , %d - %d , min_max = %d , c %d \n", step, write_loc, i, ixj, min_max, coupled_elem);
            
            arr[step * N_true + write_loc] = (i << 1) + min_max;
            arr_part[step * N_true + write_loc] = (ixj << 1) + coupled_elem;

          } else {
            write_loc = elem;
            //printf("step = %d , elem = %d , %d - %d , min_max = %d , c %d \n", step, write_loc, i, ixj, min_max, coupled_elem);
            //if (ixj > 2048 || i > 2048) printf("err \n");
            arr[step * N_true + write_loc] = (i << 1) + min_max;
            arr_part[step * N_true + write_loc] = (ixj << 1) + coupled_elem;
            elem++;
          }




        }
      }    

      step++;
    }
  }
  //printf("max size  = %d \n" , step * N_true + N_true);
  //printf(" test sort[%d] = %d , %d \n", 135968, arr[135968], arr_part[135968]);
  checkCudaErrors(cudaMemcpy(d_arr, arr, sizeof(int)*copy_size, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_arr_part, arr_part, sizeof(int)*copy_size, cudaMemcpyHostToDevice));

  //free(arr);
  //free(arr_part);
  //free(tracker);
  
}









void gpu_knn(int *R, int *C, float *V, int *G_Id, int M, int leaves, int k, float *knn, int *knn_Id, int max_nnz){
 
	int ppl = M/leaves;
	int m = 8192 / max_nnz;
  
  //int tmp = sqrt(ppl);
  //printf("tmp %d \n", tmp);
  //m = min(m, tmp);
  m = min(m, ppl);
  
  if (m > 32){ 
    m = 32; 
  }
 


  size_t free, total;
  cudaMemGetInfo(&free, &total);
  int log_size = log2(free / (sizeof(float)));
  double arr_len = pow(2, log_size); 

  float del_t1;
  float del_t2;
  cudaEvent_t t0; 
  cudaEvent_t t1;
  cudaEvent_t t2;
  int blocks = k * k;
  
  int num_blocks_tri = ppl / k;
 
  // threads for compute norms
   
  int t_b = (ppl > 1024) ? 1024 : ppl;
  int num_batch_norm = (ppl > 1024) ? ppl/1024 : 1;
  
  dim3 dimBlock_norm(t_b, 1);	
  dim3 dimGrid_norm(num_batch_norm, leaves); 

  printf("block Norms = (%d , %d) \n ", t_b, 1);
  printf("Grid Norms = (%d , %d) \n ", num_batch_norm, leaves);

  
  float *d_Norms;


  int size_tri = k;
  int blockDim_tri = size_tri * size_tri;
  if (blockDim_tri > 1024) blockDim_tri = 1024;

  dim3 dimBlock_tri(blockDim_tri, 1);	
  dim3 dimGrid_tri(num_blocks_tri, leaves); 
  
  printf("block TriPart = (%d , %d) \n ", blockDim_tri, 1);
  printf("Grid TriPart = (%d , %d) \n ", num_blocks_tri, leaves);

  
  dim3 dimGrid_sq(k, leaves);

  printf("Grid RecPart = (%d , %d) \n ", k, leaves);


  
  
  
  

  // vertical sorting 
  //dim3 dimBlock_v(k, 1);
  /*
  bool optimal_sort = true;
  int *d_sort_arr_v, *d_sort_arr_part_v;
  int *sort_arr_v, *sort_arr_part_v;
  //int steps_v;
  //if (optimal_sort) {
  
  int size_v = 2 * k;
  int blocksize_v = k;
  
  int steps_v = log2(size_v) * (log2(size_v) + 1) /2;
  sort_arr_v = (int *)malloc(sizeof(int) * steps_v * size_v);
  sort_arr_part_v = (int *)malloc(sizeof(int) * steps_v * size_v);
  
  precomp_arbsize_sortId(sort_arr_v, sort_arr_part_v, size_v, size_v, steps_v, size_v);
  


  checkCudaErrors(cudaMemcpy(d_sort_arr_v, sort_arr_v, sizeof(int)* steps_v * size_v, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_sort_arr_part_v, sort_arr_part_v, sizeof(int)* steps_v * size_v, cudaMemcpyHostToDevice));
  dim3 dimBlock_v(blocksize_v, 1);
  */
  
  int size_v = 2 * k;
  dim3 dimBlock_v(size_v, 1);
  /*
  } else {
    
    
  }
  */

  


  
  //int *d_block_indices;
  
  //checkCudaErrors(cudaMalloc((void **) &d_block_indices, sizeof(int) * 2 * num_blocks_sq));
  //par_block_indices(num_blocks_tri, d_block_indices);

  int *d_arr, *d_arr_part;

  int n_s = log2(ppl) *(log2(ppl)+1) /2;
  
  int copy_size = (ppl) * n_s;
  printf("max size = %d \n", copy_size);
  checkCudaErrors(cudaMalloc((void **) &d_arr, sizeof(int) * copy_size));
  checkCudaErrors(cudaMalloc((void **) &d_arr_part, sizeof(int) * copy_size));
  //compute_sort_ind(d_arr, d_arr_part, 8); 
  //compute_sort_ind(d_arr, d_arr_part, size); 
  checkCudaErrors(cudaMemset(d_arr, 0, sizeof(int) * copy_size));
  checkCudaErrors(cudaMemset(d_arr_part, 0, sizeof(int) * copy_size));


  printf("# leaves : %d \n", leaves);
  printf("# points/leaf : %d \n", ppl);
  printf("  max_nnz : %d \n", max_nnz); 
  
  
  

  printf(" # points = %d \n" , M);

   
  int *d_temp_knnId;
  float * d_temp_knn;
  checkCudaErrors(cudaMalloc((void **) &d_Norms, sizeof(float) * ppl * leaves));
  checkCudaErrors(cudaMalloc((void **) &d_temp_knn, sizeof(float) * ppl * leaves * m));
  //checkCudaErrors(cudaMalloc((void **) &d_temp_knnId, sizeof(int) * ppl * leaves * m));

  int blocksize, steps;
  dim3 Block_GId(1024);
  
  dim3 Grid_GId(leaves);

  checkCudaErrors(cudaEventCreate(&t0));
  checkCudaErrors(cudaEventCreate(&t1));
  checkCudaErrors(cudaEventCreate(&t2));
  checkCudaErrors(cudaEventRecord(t0, 0));

  sort_GIds <<< Grid_GId, Block_GId >>> (G_Id, ppl);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaEventRecord(t2, 0));
  checkCudaErrors(cudaEventElapsedTime(&del_t2, t0, t2));
  printf("\n Elapsed time (s) : %.4f \n ", del_t2/1000);

  compute_norm <<< dimGrid_norm, dimBlock_norm >>>(R, C, V, G_Id, d_Norms, ppl);
  
  knn_kernel_tri <<< dimGrid_tri, dimBlock_tri >>>(R, C, V, G_Id, d_Norms, k, knn, knn_Id, ppl, max_nnz, m, d_arr, d_arr_part, n_s);
  checkCudaErrors(cudaDeviceSynchronize());
  

  for (int blockInd = 0; blockInd < num_blocks_tri - 1; blockInd++){
    
    int size_part = ppl - blockInd *k;
    int blocksize = ceil(size_part/2);
    
    //while (blocksize > 1024) blocksize = ceil(blocksize/2);
    
    int N_pow2 = pow(2, ceil(log2(2 * blocksize)));
    steps = log2(N_pow2) * (log2(N_pow2) +1)/2;
    
    
    int real_size = 2 * blocksize;
    //printf("blockind = %d  , part_size = %d , N_pow2 = %d ,  blocksize = %d , realsize = %d, steps = %d \n", blockInd, size_part, blocksize, real_size, steps);
    precomp_arbsize_sortId(d_arr, d_arr_part, real_size, N_pow2, steps, copy_size);

    //int size = ceil(blocksize /2);
    dim3 dimBlock_sq(1024, 1);
    //printf("block RecPart (%d,%d) \n", blocksize, 1);
    
    int size_v = ppl - (blockInd + 1) * k;
    
    dim3 dimGrid_v(size_v, leaves);
    
    //printf("Grid VertSort (%d,%d) \n", size_v, leaves);
    
    
    
    //knn_kernel_sq <<< dimGrid_sq, dimBlock_sq >>>(R, C, V, G_Id, d_Norms, k, knn, knn_Id, ppl, max_nnz, m ,blockInd, d_arr, d_arr_part, steps, d_temp_knn, d_temp_knnId);
    knn_kernel_sq <<< dimGrid_sq, dimBlock_sq >>>(R, C, V, G_Id, d_Norms, k, knn, knn_Id, ppl, max_nnz, m ,blockInd, d_arr, d_arr_part, steps, d_temp_knn);
    checkCudaErrors(cudaDeviceSynchronize());
    //knn_kernel_v <<< dimGrid_v, dimBlock_v >>> (knn, knn_Id, k, m, ppl, blockInd, d_temp_knn, d_temp_knnId);
    knn_kernel_v <<< dimGrid_v, dimBlock_v >>> (knn, knn_Id, k, m, ppl, blockInd, d_temp_knn, G_Id);
    //knn_kernel_v_reduced <<< dimGrid_v, dimBlock_v >>> (knn, knn_Id, k, m, ppl, blockInd, d_temp_knn, d_temp_knnId, d_sort_arr_v, d_sort_arr_part_v, steps_v);
    
    checkCudaErrors(cudaDeviceSynchronize());
  } 
 
  checkCudaErrors(cudaDeviceSynchronize());
  //size_t free, total;
  cudaMemGetInfo(&free, &total);
  std::cout<<"Free memory : "<<free<<" from : "<< total <<std::endl;

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
    int M = 4096*2048;
    int leaves = 2048;
    d = 100000;
    int k = 32;
    nnzperrow = 32;
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
        if (i == 63) printf("R[%d] = %d , C[%d] = %d , V[%d] = %.4f \n", i ,h_R[i], h_R[i]+j, h_C[h_R[i] + j], h_R[i]+j, h_V[h_R[i]+j]);
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
