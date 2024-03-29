
#define SM_SIZE_1 1024
#define SM_SIZE_2 2048
#define SM_SIZE_SORT 8192

#include "FIKNN_dense.h"

__global__ void FIKNN_compute_norm_dense(float* data, int* G_Id, float* Norms, int ppl, int d) {

  int row = threadIdx.x + blockIdx.x * blockDim.x;
  int leaf_id_g = blockIdx.y;
  
  int g_rowId = leaf_id_g * ppl + row;

  int g_Id = G_Id[g_rowId];
  
  
  float norm_i = 0.0;
  
  for (int n_i = 0; n_i < d; n_i++) norm_i += data[g_Id * d + n_i] * data[g_Id * d + n_i];
  int ind_write = leaf_id_g * ppl + row;
  Norms[ind_write] = norm_i;

}

__global__ void FIKNN_tri_dense(float* data, int* G_Id, float* Norms , int k_nn, float* KNN_dist, int* KNN_Id, int ppl, int d){

  //__shared__ int SM[12000];
  //__shared__ float SM_dist[2048];
  //__shared__ int SM_Id[2048];


  int ind = threadIdx.x;
  int leaf_id_g = blockIdx.y;
  int block = blockIdx.x;



  int size_block = k_nn * (k_nn + 1) /2;

  for (int elem = ind; elem < size_block; elem += blockDim.x){

    float tmp = -8 * elem + 4 * k_nn * (k_nn+1) - 7;
    int rowId = sqrt(tmp)/2.0 - 0.5;
    rowId = k_nn - 1 - rowId;
    int colId = elem + rowId - k_nn * (k_nn + 1) / 2 + (k_nn - rowId) * ((k_nn - rowId) + 1)/2;

    int g_rowId = leaf_id_g * ppl + block * k_nn + rowId;
    int g_colId = leaf_id_g * ppl + block * k_nn + colId;

    int perm_i = G_Id[g_rowId];
    int perm_j = G_Id[g_colId];



    float norm_ij = Norms[g_rowId] + Norms[g_colId];

    float c_tmp = 0.0;

    //  inner product

    for (int mult_elem = 0; mult_elem < d; mult_elem++) c_tmp += data[perm_i * d + mult_elem] * data[perm_j * d + mult_elem];
    
   
    c_tmp = -2 * c_tmp + norm_ij;
    c_tmp = ( c_tmp > 0) ? sqrt(c_tmp) : 0.0;

    int ind_knn = leaf_id_g * ppl * k_nn + (block * k_nn + rowId) * k_nn + colId;
    int ind_knn_T = leaf_id_g * ppl * k_nn + (block * k_nn + colId) * k_nn + rowId;
    KNN_dist[ind_knn] = c_tmp;
    KNN_Id[ind_knn] = perm_j;
    if (colId > rowId) KNN_dist[ind_knn_T] = c_tmp;
    if (colId > rowId) KNN_Id[ind_knn_T] = perm_i;
    //if (leaf_id_g == 0 && block * k_nn + rowId == 100) printf("pos = %d , (%.4f , %d ) \n", colId, c_tmp, perm_j);
  }


}


__global__ void FIKNN_kernel_A_dense(float* data, int* G_Id, float* Norms, int k_nn, float* KNN_dist, int* KNN_Id, int ppl, int d, int blockInd, int* sort_arr, int* sort_arr_part, int steps, float* d_knn_temp) {


  __shared__ float SM[SM_SIZE_1];
  __shared__ float SM_dist[SM_SIZE_2];
  __shared__ int SM_Id[SM_SIZE_2];
  __shared__ float res_dist[SM_SIZE_1];
  __shared__ int res_Id[SM_SIZE_1];
  
  int row_l = blockIdx.x;
  int leaf_id_g = blockIdx.y;
  int j = threadIdx.x;
  int size_sort = 2 * blockDim.x;
  
  //int size_sort_step = size_sort - k_nn; 
  int size_part = ppl - (k_nn) * blockInd; 

  int rowId_leaf = k_nn * blockInd + row_l;
  int g_rowId_I = leaf_id_g * ppl + rowId_leaf;
  int perm_i = G_Id[g_rowId_I];
  
  float norm_i = Norms[g_rowId_I];
  //int nnz_i = ind1_i - ind0_i;


  //for (int n_i = j; n_i< d; n_i += blockDim.x) SM[n_i] = data[perm_i * d + n_i];
   
  int num_batches = size_part / (size_sort);
  __syncthreads();

  for (int col_batch = 0; col_batch < num_batches; col_batch++){
    
    
    for (int init_write = j; init_write < SM_SIZE_2; init_write += blockDim.x) SM_dist[init_write] = 1e30;

    for (int j_tmp = j; j_tmp < size_sort; j_tmp += blockDim.x){

      int colId_leaf = k_nn * (blockInd) + col_batch * size_sort + j_tmp;
      
      
      
      if (colId_leaf >= k_nn * (blockInd+1) && colId_leaf < ppl){
        
        int g_rowId_J = leaf_id_g * ppl + colId_leaf;
    
        int perm_j = G_Id[g_rowId_J];

        float norm_ij = norm_i + Norms[g_rowId_J];
        
        float c_tmp = 0.0;
      
        
        //inner product
        for (int pos_k = 0; pos_k < d; pos_k++) c_tmp += SM[pos_k] * data[perm_j * d + pos_k]; 
        
        c_tmp = -2 * c_tmp + norm_ij;
        c_tmp = ( c_tmp > 0) ? sqrt(c_tmp) : 0.0;

        SM_dist[j_tmp] = c_tmp;
        SM_Id[j_tmp] = perm_j;
        
        int size_tmp = size_part - k_nn;
        int ind_tmp = leaf_id_g * k_nn * size_tmp + row_l * size_tmp + colId_leaf - (k_nn) * (blockInd+1);
        
        d_knn_temp[ind_tmp] = c_tmp; //SM_dist[j_tmp];
        
        
        //__syncthreads();
      } else {
         
        SM_dist[j_tmp] = (j_tmp < k_nn) ? KNN_dist[leaf_id_g * ppl * k_nn + rowId_leaf * k_nn + j_tmp] : 1e30;
        SM_Id[j_tmp] = (j_tmp < k_nn) ? KNN_Id[leaf_id_g * ppl * k_nn + rowId_leaf * k_nn + j_tmp] : 0;
        
        
      }
      
    
      
      
    }
    
    __syncthreads();
    
    
    // sort and merge

    float tmp_f;
    int tmp_i;
    int ind_sort;

    __syncthreads();
    
    for (int step = 0 ; step < steps; step++){
        
      ind_sort = step * 2 * blockDim.x + j;
      
      int tid = sort_arr[ind_sort];
      int ixj = sort_arr_part[ind_sort];
      
      int min_max = (1 & tid);
      int coupled_flag = (1 & ixj);
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
      __syncthreads();
    
    }
    
   
    
    
    if (j < k_nn){
      //if (leaf_id_g == 0 && rowId_leaf == 100) printf(" pos = %d , (%.4f, %d) \n", j, SM_dist[j], SM_Id[j]);   
      res_dist[col_batch * k_nn + j] = SM_dist[j];
      res_Id[col_batch * k_nn + j] = SM_Id[j];
      
    }
    
    
    


  }

  
  // sort over results 

  
  int num_batch = size_part / size_sort;
  
  size_sort = num_batch * k_nn;
  
  
  //int part_cov = size_part / size_sort;
  //int rem_part = size_part - part_cov * size_sort_step;
  

  // implement bitonic sort 
  
  float tmp_f;
  int tmp_i;
  __syncthreads();
  
  for (int g = 2; g <= size_sort; g *= 2){
    for (int l = g/2; l>0; l /= 2){
      int ixj = j ^ l;
      if (j < size_sort){      
      if (ixj > j){
        if ((j & g) == 0){
          if (res_dist[j] > res_dist[ixj]){

            tmp_f = res_dist[ixj];
            res_dist[ixj] = res_dist[j];
            res_dist[j] = tmp_f;

            tmp_i = res_Id[ixj];
            res_Id[ixj] = res_Id[j];
            res_Id[j] = tmp_i;
          }
        } else {
          if (res_dist[j] < res_dist[ixj]){

            tmp_f = res_dist[ixj];
            res_dist[ixj] = res_dist[j];
            res_dist[j] = tmp_f;

            tmp_i = res_Id[ixj];
            res_Id[ixj] = res_Id[j];
            res_Id[j] = tmp_i;
                
          }
        }
      }
    }
    __syncthreads();
    }
  }
  
    
  
  if (j < k_nn){
    int ind_knn = leaf_id_g * ppl * k_nn + rowId_leaf * k_nn + j;
    
    KNN_dist[ind_knn] = res_dist[j];
    KNN_Id[ind_knn] = res_Id[j];
    
  }


}

/*
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


*/
__global__ void FIKNN_kernel_B_dense(float* KNN, int* KNN_Id, int k_nn, int ppl, int blockInd, float* d_temp_knn, int* G_Id){

  __shared__ float SM_dist[SM_SIZE_1];
  __shared__ int SM_Id[SM_SIZE_1];


  int tid = threadIdx.x;

  int col = blockIdx.x;
  int leaf_id_g = blockIdx.y;
  int colId_leaf = col + k_nn * (blockInd + 1);
  int size_part = ppl - (blockInd + 1) * (k_nn);
  
  if (tid < k_nn){
    int ind_tmp = leaf_id_g * k_nn * size_part + tid * size_part + col;
    SM_dist[tid] = d_temp_knn[ind_tmp];
    int rowId_g = leaf_id_g * ppl + k_nn * blockInd + tid;  
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
  
  if (tid < k_nn) {
    int ind_knn = leaf_id_g * ppl * k_nn + colId_leaf * k_nn + tid;
    KNN[ind_knn] = SM_dist[tid];
    KNN_Id[ind_knn] = SM_Id[tid];
    
  
  }

}

/*
__global__ void sort_GIds(int* G_Id, int ppl){

  int tid = threadIdx.x;
  
  int leaf_id_g = blockIdx.x;
  

  
  __shared__ int SM_GId[SM_SIZE_SORT];

  for (int tid_seq = tid; tid_seq < ppl; tid_seq += blockDim.x) SM_GId[tid_seq] = G_Id[leaf_id_g * ppl + tid_seq];
  
  __syncthreads();
  
  
  int tmp_i;


  for (int g = 2; g <= ppl; g *= 2){
    for (int l = g/2; l > 0; l /= 2){
      for (int tid_seq = tid; tid_seq < ppl; tid_seq += blockDim.x){
        
        int ixj = tid_seq ^ l;
        
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
  for (int tid_seq = tid; tid_seq < ppl; tid_seq += blockDim.x) G_Id[leaf_id_g * ppl + tid_seq] = SM_GId[tid_seq];
    
  

}

*/


/*
__global__ void test_sort(int* sort_arr, int* sort_arr_part, int N_true, int N_pow2, int steps, float* list){


  int tid = threadIdx.x;
  
  curandState_t state;
  
  
  
  
  
  
  for (int tmp = tid; tmp < N_pow2; tmp++) {
    if (tmp < N_true) *list[tmp] = curand(&state) % RAND_MAX;
    if (tmp >= N_true) list[tmp] = 1e30;
  }
  

  


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


*/



void precomp_arbsize_sortId_dense(int* d_arr, int* d_arr_part, int N_true, int N_pow2, int steps, int copy_size){

  
  
  int min_max, elem, coupled_elem;
  int loc_len = ceil(N_true/2);

  //printf("N_true = %d / %d , %d , %d, \n", N_true, N_pow2, steps, copy_size);
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
  checkCudaErrors(cudaMemcpy(d_arr, arr, sizeof(int)*copy_size, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_arr_part, arr_part, sizeof(int)*copy_size, cudaMemcpyHostToDevice));
  
}




void FIKNN_gpu_dense(float *data, int *G_Id, int M, int leaves, int k, float *knn, int *knn_Id, int d){
 
	int ppl = M/leaves;
  
  //int tmp = sqrt(ppl);
  //printf("tmp %d \n", tmp);
  //m = min(m, tmp);
  //m = min(m, ppl);
  
 


  
  
  


  float del_t1;
  cudaEvent_t t0; 
  cudaEvent_t t1;
  //int blocks = k * k;
  
  int num_blocks_tri = ppl / k;
 
  // threads for compute norms
   
  int t_b = (ppl > SM_SIZE_1) ? SM_SIZE_1 : ppl;
  int num_batch_norm = (ppl > SM_SIZE_1) ? ppl/SM_SIZE_1 : 1;
  
  dim3 dimBlock_norm(t_b, 1);	
  dim3 dimGrid_norm(num_batch_norm, leaves); 

  printf("block Norms = (%d , %d) \n ", t_b, 1);
  printf("Grid Norms = (%d , %d) \n ", num_batch_norm, leaves);

  
  float *d_Norms;


  int size_tri = (k > 32) ? 32 : k;
  int blockDim_tri = size_tri * (size_tri+1) /2;
  if (blockDim_tri > SM_SIZE_1) blockDim_tri = SM_SIZE_1;

  dim3 dimBlock_tri(blockDim_tri, 1);	
  dim3 dimGrid_tri(num_blocks_tri, leaves); 
  
  printf("block TriPart = (%d , %d) \n ", blockDim_tri, 1);
  printf("Grid TriPart = (%d , %d) \n ", num_blocks_tri, leaves);

  
  dim3 dimGrid_sq(k, leaves);

  printf("Grid RecPart = (%d , %d) \n ", k, leaves);  

  // efficient sort for kernel A
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
    
    

  int *d_arr, *d_arr_part;

  int n_s = log2(SM_SIZE_2) *(log2(SM_SIZE_2)+1) /2;
  
  int copy_size = (ppl) * n_s;

  size_t free, total, m1, m2, m3;

  cudaMemGetInfo(&free, &total);
  checkCudaErrors(cudaMalloc((void **) &d_arr, sizeof(int) * copy_size));
  checkCudaErrors(cudaMalloc((void **) &d_arr_part, sizeof(int) * copy_size));

  
  checkCudaErrors(cudaMemset(d_arr, 0, sizeof(int) * copy_size));
  checkCudaErrors(cudaMemset(d_arr_part, 0, sizeof(int) * copy_size));
  cudaMemGetInfo(&m1, &total);


   
  float * d_temp_knn;
  checkCudaErrors(cudaMalloc((void **) &d_Norms, sizeof(float) * ppl * leaves));
  
  cudaMemGetInfo(&m2, &total);
  checkCudaErrors(cudaMalloc((void **) &d_temp_knn, sizeof(float) * ppl * leaves * k));
  cudaMemGetInfo(&m3, &total);

  int steps;
  dim3 Block_GId(SM_SIZE_1);
  
  dim3 Grid_GId(leaves);

  checkCudaErrors(cudaEventCreate(&t0));
  checkCudaErrors(cudaEventCreate(&t1));
  
  checkCudaErrors(cudaEventRecord(t0, 0));

  //sort_GIds_dense <<< Grid_GId, Block_GId >>> (G_Id, ppl);
  checkCudaErrors(cudaDeviceSynchronize());
  
  FIKNN_compute_norm_dense <<< dimGrid_norm, dimBlock_norm >>>(data, G_Id, d_Norms, ppl, d);
  checkCudaErrors(cudaDeviceSynchronize());
  FIKNN_tri_dense <<< dimGrid_tri, dimBlock_tri >>>(data, G_Id, d_Norms, k, knn, knn_Id, ppl, d);
  checkCudaErrors(cudaDeviceSynchronize());
  
  
  for (int blockInd = 0; blockInd < num_blocks_tri - 1; blockInd++){
    
    int size_part = ppl - blockInd *k;
    int blocksize = ceil(size_part/2);
    
    while (blocksize > SM_SIZE_1) blocksize = ceil(blocksize/2);
    
    int N_pow2 = pow(2, ceil(log2(2 * blocksize)));
    steps = log2(N_pow2) * (log2(N_pow2) +1)/2;
    
    
    int real_size = 2 * blocksize;
    
    precomp_arbsize_sortId_dense(d_arr, d_arr_part, real_size, N_pow2, steps, copy_size);

    
    dim3 dimBlock_sq(blocksize, 1);

    
    
    int size_v = ppl - (blockInd + 1) * k;
    
    dim3 dimGrid_v(size_v, leaves);
    //printf("blockInd = %d , size_part = %d , blockSize = %d , N_pow2 = %d , steps = %d \n", blockInd, size_part, blocksize, N_pow2, steps); 
    FIKNN_kernel_A_dense <<< dimGrid_sq, dimBlock_sq >>>(data, G_Id, d_Norms, k, knn, knn_Id, ppl, d ,blockInd, d_arr, d_arr_part, steps, d_temp_knn);
    checkCudaErrors(cudaDeviceSynchronize());
    
    FIKNN_kernel_B_dense <<< dimGrid_v, dimBlock_v >>> (knn, knn_Id, k, ppl, blockInd, d_temp_knn, G_Id);
    
    checkCudaErrors(cudaDeviceSynchronize());
  } 
 
  checkCudaErrors(cudaDeviceSynchronize());
  

  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaEventRecord(t1, 0));
  checkCudaErrors(cudaEventSynchronize(t1));
  checkCudaErrors(cudaEventElapsedTime(&del_t1, t0, t1));
 
  printf("\n Elapsed time (s) : %.4f \n ", del_t1/1000);

  printf("\n Memory for sort %.4f GB \n", (free-m1)/1e9);
  printf("\n Memory for norm %.4f GB \n", (m1-m2)/1e9);
  printf("\n Memory for temp storage %.4f GB \n", (m2-m3)/1e9);
 
  checkCudaErrors(cudaFree(d_Norms));
  checkCudaErrors(cudaEventDestroy(t0));
  checkCudaErrors(cudaEventDestroy(t1));

}









