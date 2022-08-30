
#define SM_SIZE_1 1024
#define SM_SIZE_2 2048
#define SM_SIZE_SORT 8192

#include "FIKNN_sparse.h"

__global__ void FIKNN_compute_norm(float* data, int* G_Id, float* Norms, int ppl, int dim) {

  int row = threadIdx.x + blockIdx.x * blockDim.x;
  int leaf_id_g = blockIdx.y;
  
  int g_rowId = leaf_id_g * ppl + row;
  int g_Id = G_Id[g_rowId];

   
  //int ind0_i = R[g_Id];
 
  //int nnz = R[g_Id + 1] - ind0_i;
  float norm_i = 0.0;
   
  for (int n_i = 0; n_i < dim; n_i += 1) {
    norm_i += data[g_Id * dim + n_i] * data[g_Id * dim + n_i];
  }
  Norms[g_Id] = norm_i;
}

__global__ void knn_kernel_tri(float* data, int* G_Id, float* Norms , int k_nn, float* KNN_dist_tmp, int ppl, int dim, int iternum) {


  __shared__ int SM[12288];


  int ind = threadIdx.x;
  int leaf_id_g = blockIdx.y;
  int block = blockIdx.x;



  int size_block = k_nn * (k_nn + 1) /2;

  int sm_pt_j = ind / k_nn;
  int sm_pt_i = ind - k_nn * sm_pt_j;

  int sm_g_ptId = leaf_id_g * ppl + block * k_nn + sm_pt_i;

  int sm_perm_pt = G_Id[sm_g_ptId];

  int sm_ind0_i = sm_perm_pt * dim;
  //int sm_nnz = R[sm_perm_pt + 1] - sm_ind0_i;

  // TODO: optimize this section for uniform read to the SM
  int sm_shift = sm_pt_i * dim;
  if (sm_pt_j == 0){
    for (int n_i = 0; n_i < dim; n_i++) SM[sm_shift + n_i] = C[sm_ind0_i + n_i];
  }
  __syncthreads();
  
  for (int elem = ind; elem < size_block; elem += blockDim.x){

    float tmp = -8 * elem + 4 * k_nn * (k_nn+1) - 7;
    int rowId = sqrt(tmp)/2.0 - 0.5;
    rowId = k_nn - 1 - rowId;
    int colId = elem + rowId - k_nn * (k_nn + 1) / 2 + (k_nn - rowId) * ((k_nn - rowId) + 1)/2;

    int g_rowId = leaf_id_g * ppl + block * k_nn + rowId;
    int g_colId = leaf_id_g * ppl + block * k_nn + colId;

    int perm_i = G_Id[g_rowId];
    int perm_j = G_Id[g_colId];

    int ind0_i = perm_i * dim;
    //int ind1_i = R[perm_i + 1];

    int ind0_j = perm_j * dim;
    //int ind1_j = R[perm_j + 1];

    //int nnz_i = ind1_i - ind0_i;
    //int nnz_j = ind1_j - ind0_j;

    float norm_ij = Norms[perm_i] + Norms[perm_j];

    float c_tmp = 0.0;
    int tmp_0, tmp_1, ind_jk, k, ret, testInd;

    ret = 0;
    testInd = 0;

    for (int pos_k = 0; pos_k < dim; pos_k++) c_tmp += data[ind0_i + pos_k] * data[ind0_j + pos_k];
    /*
    if (nnz_i > 0 && nnz_j >0){
      for (int pos_k = 0; pos_k < nnz_j; pos_k++){
    
        k = SM[dim * colId + pos_k];
        
        // Binary search
        for (int l = nnz_i - ret; l > 1; l -= floorf(l/2.0)){
          tmp_0 = ret + l;
          tmp_1 = nnz_i - 1;
          testInd = (tmp_0 < tmp_1) ? tmp_0 : tmp_1;
          ret = (SM[testInd + max_nnz * rowId] <= k) ? testInd : ret;
        }
      
        tmp_0 = ret + 1;
        tmp_1 = nnz_i - 1;
        testInd = (tmp_0 < tmp_1 ) ? tmp_0 : tmp_1;
      
        ret = (SM[testInd + max_nnz * rowId] <= k) ? testInd : ret;
      
        ind_jk = (SM[ret + max_nnz * rowId] == k) ? ret : -1;
        c_tmp += (ind_jk != -1) ? V[ind0_j + pos_k] * V[ind0_i + ind_jk] : 0;
      
      }
    }
    */
    c_tmp = -2 * c_tmp + norm_ij;
    c_tmp = (c_tmp > 0.0) ? sqrt(c_tmp) : 0.0;

    int ind_knn = leaf_id_g * ppl * k_nn + (block * k_nn + rowId) * k_nn + colId;
    int ind_knn_T = leaf_id_g * ppl * k_nn + (block * k_nn + colId) * k_nn + rowId;
    
    //int ind_knn = leaf_id_g * ppl * k_nn + (block * k_nn + rowId) * ppl + colId;
    //int ind_knn_T = leaf_id_g * ppl * k_nn + (block * k_nn + colId) * ppl + rowId;
    
    KNN_dist_tmp[ind_knn] = (iternum >0 && colId == rowId) ? 1e30 : c_tmp;
    if (colId > rowId) KNN_dist_tmp[ind_knn_T] = c_tmp;

    int test_pt = leaf_id_g * ppl + (block * k_nn + rowId);
    int test_pt2 = leaf_id_g * ppl + (block * k_nn + colId);

    //if (G_Id[test_pt] == 0) printf("D[%d] = %.4f , Id = %d , write_at = %d , iter = %d \n", test_pt2, KNN_dist_tmp[ind_knn], G_Id[test_pt2], ind_knn, iternum); 
    //if (G_Id[test_pt2] == 0 && G_Id[test_pt] != 0) printf("D[%d] = %.4f , Id = %d , write_at = %d , iter = %d \n", test_pt, KNN_dist_tmp[ind_knn], G_Id[test_pt], ind_knn_T, iternum); 
  }
  
}

__global__ void knn_kernel_A(float* data, int* G_Id, float* Norms, int k_nn, float* KNN_dist, int* KNN_Id, int ppl, int dim, int blockInd, int* sort_arr, int* sort_arr_part, int steps, float* d_knn_temp) {


  __shared__ int SM[SM_SIZE_1];
  __shared__ float SM_dist[SM_SIZE_2];
  __shared__ int SM_Id[SM_SIZE_2];
  __shared__ float res_dist[SM_SIZE_1];
  __shared__ int res_Id[SM_SIZE_1];
  
  int row_l = blockIdx.x;
  int leaf_id_g = blockIdx.y;
  int j = threadIdx.x;
  int size_sort = 2 * blockDim.x;
  
  int size_part = ppl - (k_nn) * blockInd; 

  int rowId_leaf = k_nn * blockInd + row_l;
  int g_rowId_I = leaf_id_g * ppl + rowId_leaf;
  int perm_i = G_Id[g_rowId_I];
  int ind0_i = perm_i * dim;
  //int ind1_i = R[perm_i+1;
  //float norm_i = Norms[g_rowId_I];
  float norm_i = Norms[perm_i];
  //int nnz_i = ind1_i - ind0_i;


  for (int n_i = j; n_i< dim; n_i += blockDim.x) SM[n_i] = data[ind0_i + n_i];
   
  int num_batches = size_part / (size_sort);
  __syncthreads();

  for (int col_batch = 0; col_batch < num_batches; col_batch++){
    
    for (int init_write = j; init_write < SM_SIZE_2; init_write += blockDim.x) SM_dist[init_write] = 1e30;


    for (int j_tmp = j; j_tmp < size_sort; j_tmp += blockDim.x){
      //for (int pt = 0; pt < 2; pt++){
      //int j_tmp = 2 * j + pt;

      int colId_leaf = k_nn * (blockInd) + col_batch * size_sort + j_tmp;
      
      
      
      if (colId_leaf >= k_nn * (blockInd+1) && colId_leaf < ppl){
        
        int g_rowId_J = leaf_id_g * ppl + colId_leaf;
    
        int perm_j = G_Id[g_rowId_J];
        int ind0_j = perm_j * dim;
        //int ind1_j = R[perm_j+1];

        //int nnz_j = ind1_j - ind0_j;

        //float norm_ij = norm_i + Norms[g_rowId_J];
        float norm_ij = norm_i + Norms[perm_j];
        
        float c_tmp = 0.0;
        int tmp_0, tmp_1, ind_jk, k, ret, testInd;
      
        ret = 0;
        testInd = 0;
        float s = 0.0;
        // loop over the elements of j
        for (int pos_k = 0; pos_k < dim; pos_k++) c_tmp += data[ind0_i + pos_k] * data[ind0_j + pos_k];

        /*
        if (nnz_i >0 && nnz_j > 0){
          for (int pos_k = 0; pos_k < nnz_j; pos_k++){
          
            k = C[ind0_j + pos_k];
      
            // Binary search
      
            for (int l = nnz_i - ret; l > 1; l -= floorf(l/2.0)){
  
              tmp_0 = ret + l;
              tmp_1 = nnz_i - 1;
              
              testInd = (tmp_0 < tmp_1) ? tmp_0 : tmp_1;
              
              ret = (SM[testInd] <= k) ? testInd : ret;
            }

            tmp_0 = ret + 1;
            tmp_1 = nnz_i - 1;
            
            testInd = (tmp_0 < tmp_1 ) ? tmp_0 : tmp_1;
            ret = (SM[testInd] <= k) ? testInd : ret;
            
            ind_jk = (SM[ret] == k) ? ret : -1;
            c_tmp += (ind_jk != -1) ? V[ind0_j + pos_k] * V[ind0_i + ind_jk] : 0;

          } 
        }
        */
      
        c_tmp = -2 * c_tmp + norm_ij;
        c_tmp = ( c_tmp > 0) ? sqrt(c_tmp) : 0.0;

        SM_dist[j_tmp] = c_tmp;
        SM_Id[j_tmp] = perm_j;
        
        int size_tmp = size_part - k_nn;
        int ind_tmp = leaf_id_g * k_nn * size_tmp + row_l * size_tmp + colId_leaf - (k_nn) * (blockInd+1);
            
        d_knn_temp[ind_tmp] = c_tmp; 
        //d_knnId_temp[ind_tmp] = perm_i;
        
        
      } else {
        int ind_pt = G_Id[leaf_id_g * ppl+ rowId_leaf];
        int ind_read = ind_pt * k_nn + j_tmp;
        SM_dist[j_tmp] = (j_tmp < k_nn) ? KNN_dist[ind_read] : 1e30;
        SM_Id[j_tmp] = (j_tmp < k_nn) ? KNN_Id[ind_read] : 0;
        
      }
      
      
      //if (perm_i == 0) printf("read knn from B, D[%d] = %.4f , ind = %d \n", j_tmp, SM_dist[j_tmp], SM_Id[j_tmp]);
      
      
      
    }
    __syncthreads();
    
    for (int j_tmp = j; j_tmp < size_sort; j_tmp += blockDim.x){
      //for (int pt = 0; pt < 2; pt++){
      //int j_tmp = 2 * j + pt;

      int colId_leaf = k_nn * (blockInd) + col_batch * size_sort + j_tmp;
      if (colId_leaf >= k_nn * (blockInd+1) && colId_leaf < ppl){
        float val = SM_dist[j_tmp];
        int index = SM_Id[j_tmp];
        
        for (int ind_check=0; ind_check < k_nn; ind_check++){
          if (val == SM_dist[ind_check] && index == SM_Id[ind_check]){
            SM_dist[j_tmp] = 1e30;
            SM_Id[j_tmp] = -1;
            break;
          }
        }
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
   
    //int ind_knn = leaf_id_g * ppl * k_nn + rowId_leaf * k_nn + j;
    int ind_pt = leaf_id_g * ppl + rowId_leaf;
    int ind_pt_g = G_Id[ind_pt];
    int write_ind = ind_pt_g * k_nn + j;
    
    //if (ind_pt_g == 0) printf(" D[%d] = %.4f, ind = %d , write_ind = %d \n", j, res_dist[j], res_Id[j], write_ind);
    
    KNN_dist[write_ind] = res_dist[j];
    KNN_Id[write_ind] = res_Id[j];
    //if (ind_pt_g == 0) printf("res A , D[%d] = %.4f , id = %d , write_ind = %d \n", j,res_dist[j], res_Id[j], write_ind);
    
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
__global__ void knn_kernel_B(float* KNN, int* KNN_Id, int k_nn, int ppl, int blockInd, float* d_temp_knn, int* G_Id, bool init){

  __shared__ float SM_dist[SM_SIZE_1];
  __shared__ int SM_Id[SM_SIZE_1];

  int tid = threadIdx.x;

  int col = blockIdx.x;
  int leaf_id_g = blockIdx.y;
  
  int colId_leaf = (init) ? col : col + k_nn * (blockInd + 1);
  //int size_part = (init) ? ppl : ppl - (blockInd + 1) * (k_nn);
  int size_part = (init) ? ppl : ppl - (blockInd + 1) * (k_nn);
  
  if (tid < k_nn){
    int ind_tmp = (init) ? leaf_id_g * ppl * k_nn + col * k_nn + tid : leaf_id_g * k_nn * size_part + tid * size_part + col;
    //int ind_tmp = leaf_id_g * k_nn * size_part + tid * size_part + colId_leaf;
    SM_dist[tid] = d_temp_knn[ind_tmp];

    int block = col / k_nn;
    int rowId_g = (init) ? leaf_id_g * ppl + block * k_nn + tid : leaf_id_g * ppl + k_nn * blockInd + tid;  
    SM_Id[tid] = G_Id[rowId_g];
     
  } else {

    int ind_pt_knn = leaf_id_g * ppl + colId_leaf;
    int ind_pt_knn_g = G_Id[ind_pt_knn];

    int ind_knn = ind_pt_knn_g * k_nn + tid - k_nn;
    SM_dist[tid] = KNN[ind_knn];
    SM_Id[tid] = KNN_Id[ind_knn];
      
  }


  __syncthreads();
  
  if (tid < k_nn){
    float val = SM_dist[tid];
    int index = SM_Id[tid];
    for (int ind_check = 0; ind_check < k_nn; ind_check++){
      if (val == SM_dist[ind_check + k_nn] && index == SM_Id[ind_check + k_nn]){
        SM_dist[tid] = 1e30;
        SM_Id[tid] = -1;
        break;
      }
    }
    

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
  //__syncthreads();
  int ind_pt = leaf_id_g * ppl + colId_leaf;    
  int ind_pt_g = G_Id[ind_pt];
  int write_ind = ind_pt_g * k_nn + tid;
  
  //if (ind_pt_g == 0) printf("write B, Dist[%d] = %.4f , Id = %d , write_ind = %d \n", tid, KNN[write_ind],  KNN_Id[write_ind] , write_ind);
   
  if (tid < k_nn) {
    
    //int ind_pt = leaf_id_g * ppl * k_nn + colId_leaf * k_nn + tid;
    /*
    int ind_pt = leaf_id_g * ppl + colId_leaf;    
    int ind_pt_g = G_Id[ind_pt];
    int write_ind = ind_pt_g * k_nn + tid;
    */
    KNN[write_ind] = SM_dist[tid];
    KNN_Id[write_ind] = SM_Id[tid];
    
    //if (ind_pt_g == 0) printf("write B, Dist[%d] = %.4f , Id = %d , write_ind = %d \n", tid, KNN[write_ind],  KNN_Id[write_ind] , write_ind);
    
  
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



void precomp_arbsize_sortId(int* d_arr, int* d_arr_part, int N_true, int N_pow2, int steps, int copy_size){

  
  
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
            
            arr[step * N_true + write_loc] = (i << 1) + min_max;
            arr_part[step * N_true + write_loc] = (ixj << 1) + coupled_elem;

          } else {
            write_loc = elem;
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
  free(arr);
  free(arr_part); 
}



void FIKNN_sparse_gpu(float *V, int *G_Id, int M, int leaves, int k, float *knn, int *knn_Id, int dim, int iternum){



  int ppl = M/leaves;

  float del_t1;
  cudaEvent_t t0;
  cudaEvent_t t1;

  int num_blocks_tri = ppl / k;
  int C_len = dim * M; //R[M];
  //int max_nnz_pow2 = pow(2, ceil(log2(max_nnz)));


  int t_b = (ppl > SM_SIZE_1) ? SM_SIZE_1 : ppl;
  int num_batch_norm = (ppl > SM_SIZE_1) ? ppl/SM_SIZE_1 : 1;

  dim3 dimBlock_norm(t_b, 1);
  dim3 dimGrid_norm(num_batch_norm, leaves);

  printf("----------------------------- Start of sfiknn ----------------------------- \n\n");

  float *d_Norms;


  int size_tri = (k > 32) ? 32 : k;
  int blockDim_tri = size_tri * (size_tri + 1)/2;
  if (blockDim_tri > SM_SIZE_1) blockDim_tri = SM_SIZE_1;

  dim3 dimBlock_tri(blockDim_tri, 1);
  dim3 dimGrid_tri(num_blocks_tri, leaves);



  dim3 dimGrid_sq(k, leaves);
  printf(" number of points = %d \n", M);
  printf(" number of leaves = %d \n\n", leaves);

  printf(" dim GridThreads IterativePart = (%d , %d) \n", k, leaves);
  printf(" dim BlockThreads  Norms = (%d , %d) \n", t_b, 1);
  printf(" dim GridThreads Norms = (%d , %d) \n", num_batch_norm, leaves);
  printf(" dim BlockThreads Diagonal Distances = (%d , %d) \n", blockDim_tri, 1);
  printf(" dim GridThreads Diagonal Distances = (%d , %d) \n", num_blocks_tri, leaves);

  int size_v_block = 2 * k;
  dim3 dimBlock_v(size_v_block, 1);



  int *d_arr, *d_arr_part;

  int n_s = log2(SM_SIZE_2) *(log2(SM_SIZE_2)+1) /2;

  int copy_size = (ppl) * n_s;

  size_t free, total, m1, m2, m3;

  cudaMemGetInfo(&free, &total);
  int *d_R, *d_GId, *d_C, *d_knn_Id;
  float *d_V, *d_knn;  

  checkCudaErrors(cudaMalloc((void **) &d_GId, sizeof(int) * M));
  checkCudaErrors(cudaMalloc((void **) &d_data, sizeof(float) * C_len));

  checkCudaErrors(cudaMalloc((void **) &d_knn_Id, sizeof(int) *M*k));
  checkCudaErrors(cudaMalloc((void **) &d_knn, sizeof(float) *M*k));

  checkCudaErrors(cudaMemcpy(d_data, data, sizeof(float) * C_len, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_GId, G_Id, sizeof(int) * M, cudaMemcpyHostToDevice)); 
  checkCudaErrors(cudaMemcpy(d_knn_Id, G_Id, sizeof(int) * M, cudaMemcpyHostToDevice)); 
  checkCudaErrors(cudaMemcpy(d_knn, knn, sizeof(float) * M * k, cudaMemcpyHostToDevice)); 
  checkCudaErrors(cudaMemcpy(d_knn_Id, knn_Id, sizeof(int) * M * k, cudaMemcpyHostToDevice)); 

  checkCudaErrors(cudaMalloc((void **) &d_arr, sizeof(int) * copy_size));
  checkCudaErrors(cudaMalloc((void **) &d_arr_part, sizeof(int) * copy_size));


  checkCudaErrors(cudaMemset(d_arr, 0, sizeof(int) * copy_size));
  checkCudaErrors(cudaMemset(d_arr_part, 0, sizeof(int) * copy_size));
  cudaMemGetInfo(&m1, &total);



  float * d_temp_knn;
  checkCudaErrors(cudaMalloc((void **) &d_Norms, sizeof(float) * M));

  cudaMemGetInfo(&m2, &total);
  checkCudaErrors(cudaMalloc((void **) &d_temp_knn, sizeof(float) * M * k));
  cudaMemGetInfo(&m3, &total);

  int steps;
  //dim3 Block_GId(SM_SIZE_1);

  //dim3 Grid_GId(leaves);

  checkCudaErrors(cudaEventCreate(&t0));
  checkCudaErrors(cudaEventCreate(&t1));

  checkCudaErrors(cudaEventRecord(t0, 0));

  //sort_GIds <<< Grid_GId, Block_GId >>> (G_Id, ppl);
  //checkCudaErrors(cudaDeviceSynchronize());

  FIKNN_compute_norm <<< dimGrid_norm, dimBlock_norm >>>(d_data, d_GId, d_Norms, ppl, dim);
  checkCudaErrors(cudaDeviceSynchronize());

  
  knn_kernel_tri <<< dimGrid_tri, dimBlock_tri >>>(d_data, d_GId, d_Norms, k, d_temp_knn, ppl, dim, iternum);
  checkCudaErrors(cudaDeviceSynchronize());

  int size_v = ppl;
  dim3 dimGrid_v(size_v, leaves);
  
  knn_kernel_B <<< dimGrid_v, dimBlock_v >>> (d_knn, d_knn_Id, k, ppl, 0, d_temp_knn, d_GId, true);
  checkCudaErrors(cudaDeviceSynchronize());
  
  for (int blockInd = 0; blockInd < num_blocks_tri - 1; blockInd++){

    int size_part = ppl - blockInd *k;
    int blocksize = ceil(size_part/2);

    while (blocksize > SM_SIZE_1) blocksize = ceil(blocksize/2);

    int N_pow2 = pow(2, ceil(log2(2 * blocksize)));
    steps = log2(N_pow2) * (log2(N_pow2) +1)/2;


    steps = log2(N_pow2) * (log2(N_pow2) +1)/2;


    int real_size = 2 * blocksize;

    precomp_arbsize_sortId(d_arr, d_arr_part, real_size, N_pow2, steps, copy_size);


    dim3 dimBlock_sq(blocksize, 1);



    int size_v2 = ppl - (blockInd + 1) * k;
    dim3 dimGrid_v2(size_v2, leaves);

    knn_kernel_A <<< dimGrid_sq, dimBlock_sq >>>(d_data, d_GId, d_Norms, k, d_knn, d_knn_Id, ppl, dim, blockInd, d_arr, d_arr_part, steps, d_temp_knn);
    checkCudaErrors(cudaDeviceSynchronize());

    knn_kernel_B <<< dimGrid_v2, dimBlock_v >>> (d_knn, d_knn_Id, k, ppl, blockInd, d_temp_knn, d_GId, false);

    checkCudaErrors(cudaDeviceSynchronize());
  }
  
  checkCudaErrors(cudaDeviceSynchronize());


  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaEventRecord(t1, 0));
  checkCudaErrors(cudaEventSynchronize(t1));
  checkCudaErrors(cudaEventElapsedTime(&del_t1, t0, t1));
  checkCudaErrors(cudaMemcpy(knn, d_knn, sizeof(float) * M * k, cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(knn_Id, d_knn_Id, sizeof(int) * M * k, cudaMemcpyDeviceToHost));
  

  checkCudaErrors(cudaFree(d_Norms));
  checkCudaErrors(cudaFree(d_R));
  checkCudaErrors(cudaFree(d_C));
  checkCudaErrors(cudaFree(d_V));
  checkCudaErrors(cudaFree(d_GId));
  checkCudaErrors(cudaFree(d_knn_Id));
  checkCudaErrors(cudaFree(d_knn));
  checkCudaErrors(cudaFree(d_temp_knn));
  checkCudaErrors(cudaFree(d_arr_part));
  checkCudaErrors(cudaFree(d_arr));

  checkCudaErrors(cudaEventDestroy(t0));
  checkCudaErrors(cudaEventDestroy(t1));
  cudaMemGetInfo(&free, &total);
  //printf("\n Memory free end %.4f GB \n", (total-free)/1e9);
  printf(" Elapsed time (s) : %.4f \n\n", del_t1/1000);

  printf(" Memory : storing norms = %.4f GB \n", (m1-m2)/1e9);
  printf(" Memory : precomputing the sort indices = %.4f GB \n", (free-m1)/1e9);
	printf(" Memory : temporary storage = %.4f GB \n", (m2-m3)/1e9);
  
  printf("----------------------------- End of the sfiknn -----------------------------\n\n");


}











