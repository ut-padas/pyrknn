
#define SM_SIZE_1 1024
#define SM_SIZE_2 2048
#define SM_SIZE_SORT 8192

#include "FIKNN_sparse.h"

__global__ void FIKNN_compute_norm(int* R, int* C, float* V, int* G_Id, float* Norms, int ppl) {

  //int row = threadIdx.x + blockIdx.x * blockDim.x;
  int ind = threadIdx.x;
  int leaf_id_g = blockIdx.z * blockDim.y + blockIdx.y;
  for (int row = ind; row < ppl; row += blockDim.x){
    int g_rowId = leaf_id_g * ppl + row;
    //changed
  
    int g_Id = g_rowId;

   
    int ind0_i = R[g_Id];
 
    int nnz = R[g_Id + 1] - ind0_i;
    float norm_i = 0.0;
   
    for (int n_i = 0; n_i < nnz; n_i += 1) {
      norm_i += V[ind0_i + n_i] * V[ind0_i + n_i];
    }
    Norms[g_Id] = norm_i;
  }
}

__global__ void knn_kernel_tri(int* R, int* C, float* V, int* G_Id, float* Norms , int k_nn, float* KNN_dist_tmp, int ppl, int max_nnz, int iternum) {



  int ind = threadIdx.x;
  int leaf_id_g = blockIdx.z * blockDim.y + blockIdx.y;
  int block = blockIdx.x;



  int size_block = k_nn * (k_nn + 1) /2;
  
  for (int elem = ind; elem < size_block; elem += blockDim.x){

    float tmp = -8 * elem + 4 * k_nn * (k_nn+1) - 7;
    int rowId = sqrt(tmp)/2.0 - 0.5;
    rowId = k_nn - 1 - rowId;
    int colId = elem + rowId - k_nn * (k_nn + 1) / 2 + (k_nn - rowId) * ((k_nn - rowId) + 1)/2;

    float c_tmp = 0.0;

    int g_rowId = leaf_id_g * ppl + block * k_nn + rowId;
    int g_colId = leaf_id_g * ppl + block * k_nn + colId;
   
    //changed 
    int perm_i = g_rowId;
    int perm_j = g_colId;

    int ind0_i = R[perm_i];
    int ind1_i = R[perm_i + 1];

    int ind0_j = R[perm_j];
    int ind1_j = R[perm_j + 1];

    int nnz_i = ind1_i - ind0_i;
    int nnz_j = ind1_j - ind0_j;

    float norm_ij = Norms[perm_i] + Norms[perm_j];

    int tmp_0, tmp_1, ind_jk, k, ret, testInd;

    ret = 0;
    testInd = 0;


    if (nnz_i > 0 && nnz_j >0){
      for (int pos_k = 0; pos_k < nnz_j; pos_k++){
    
        k = C[ind0_j + pos_k];
        
        // Binary search
        for (int l = nnz_i - ret; l > 1; l -= floorf(l/2.0)){
          tmp_0 = ret + l;
          tmp_1 = nnz_i - 1;
          testInd = (tmp_0 < tmp_1) ? tmp_0 : tmp_1;
          ret = (C[ind0_i + testInd] <= k) ? testInd : ret;
        }
      
        tmp_0 = ret + 1;
        tmp_1 = nnz_i - 1;
        testInd = (tmp_0 < tmp_1 ) ? tmp_0 : tmp_1;
      
        ret = (C[testInd + ind0_i] <= k) ? testInd : ret;
      
        ind_jk = (C[ret + ind0_i] == k) ? ret : -1;
        c_tmp += (ind_jk != -1) ? V[ind0_j + pos_k] * V[ind0_i + ind_jk] : 0;
      
      }
    }
    c_tmp = -2 * c_tmp + norm_ij;
    c_tmp = (c_tmp > 2e-6) ? sqrt(c_tmp) : 0.0;
    
    // changed 
    int gid_pt = leaf_id_g * ppl + block * k_nn + rowId;
    int gid_pt_T = leaf_id_g * ppl + block * k_nn + colId;
    int ind_knn = gid_pt * k_nn + colId;
    int ind_knn_T = gid_pt_T * k_nn + rowId;
    KNN_dist_tmp[ind_knn] = (iternum >0 && colId == rowId) ? 1e30 : c_tmp;
    if (colId > rowId) KNN_dist_tmp[ind_knn_T] = c_tmp;
    
  }
  
}



__global__ void knn_kernel_tri_last(int* R, int* C, float* V, int* G_Id, float* Norms , int k_nn, float* KNN_dist_tmp, int ppl, int max_nnz, int iternum, int rem_len , int blockId) {




  int ind = threadIdx.x;
  int leaf_id_g = blockIdx.z * blockDim.y + blockIdx.y;
  int block = blockId;
  


  int size_block = rem_len * (rem_len + 1) /2;


  for (int elem = ind; elem < size_block; elem += blockDim.x){

    float tmp = -8 * elem + 4 * rem_len * (rem_len+1) - 7;
    int rowId = sqrt(tmp)/2.0 - 0.5;
    rowId = rem_len - 1 - rowId;
    int colId = elem + rowId - rem_len * (rem_len + 1) / 2 + (rem_len - rowId) * ((rem_len - rowId) + 1)/2;

    float c_tmp = 0.0;
    if (block * k_nn + rowId < ppl && block * k_nn + colId < ppl){

    int g_rowId = leaf_id_g * ppl + block * k_nn + rowId;
    int g_colId = leaf_id_g * ppl + block * k_nn + colId;

    //changed
    int perm_i = g_rowId;
    int perm_j = g_colId;

    int ind0_i = R[perm_i];
    int ind1_i = R[perm_i + 1];

    int ind0_j = R[perm_j];
    int ind1_j = R[perm_j + 1];

    int nnz_i = ind1_i - ind0_i;
    int nnz_j = ind1_j - ind0_j;

    float norm_ij = Norms[perm_i] + Norms[perm_j];

    int tmp_0, tmp_1, ind_jk, k, ret, testInd;

    ret = 0;
    testInd = 0;


    if (nnz_i > 0 && nnz_j >0){
      for (int pos_k = 0; pos_k < nnz_j; pos_k++){

        k = C[ind0_j + pos_k];

        // Binary search
        for (int l = nnz_i - ret; l > 1; l -= floorf(l/2.0)){
          tmp_0 = ret + l;
          tmp_1 = nnz_i - 1;
          testInd = (tmp_0 < tmp_1) ? tmp_0 : tmp_1;
          ret = (C[ind0_i + testInd] <= k) ? testInd : ret;
        }

        tmp_0 = ret + 1;
        tmp_1 = nnz_i - 1;
        testInd = (tmp_0 < tmp_1 ) ? tmp_0 : tmp_1;

        ret = (C[testInd + ind0_i] <= k) ? testInd : ret;

        ind_jk = (C[ret + ind0_i] == k) ? ret : -1;
        c_tmp += (ind_jk != -1) ? V[ind0_j + pos_k] * V[ind0_i + ind_jk] : 0;

      }
    }
    c_tmp = -2 * c_tmp + norm_ij;
    c_tmp = (c_tmp > 2e-6) ? sqrt(c_tmp) : 0.0;

    } else {
      c_tmp = 1e30;
    }


    // changed
    int gid_pt = leaf_id_g * ppl + block * k_nn + rowId;
    int gid_pt_T = leaf_id_g * ppl + block * k_nn + colId;
    int ind_knn = gid_pt * k_nn + colId;
    int ind_knn_T = gid_pt_T * k_nn + rowId;

    KNN_dist_tmp[ind_knn] = (iternum >0 && colId == rowId) ? 1e30 : c_tmp;
    if (colId > rowId) KNN_dist_tmp[ind_knn_T] = c_tmp;

    for (int row_tmp = 0; row_tmp<rem_len; row_tmp++){
      for (int q = ind + rem_len; q < k_nn; q += blockDim.x){
        gid_pt = leaf_id_g * ppl + block * k_nn + row_tmp;
        ind_knn = gid_pt * k_nn + q;
        KNN_dist_tmp[ind_knn] = 1e30;
      } 
    } 


  }

}

__global__ void knn_kernel_A(int* R, int* C, float* V, int* G_Id, float* Norms, int k_nn, float* KNN_dist, int* KNN_Id, int ppl, int max_nnz, int blockInd, int* sort_arr, int* sort_arr_part, int steps, float* d_knn_temp) {


  __shared__ int SM[500];
  __shared__ float SM_dist[SM_SIZE_2];
  __shared__ int SM_Id[SM_SIZE_2];
  
  int row_l = blockIdx.x;
  int leaf_id_g = blockIdx.z * blockDim.y + blockIdx.y;
  int j = threadIdx.x;
  
  int size_part = ppl - (k_nn) * blockInd; 
  float size_sort_f = size_part;
  int num_batches = 1;
  while (size_sort_f > SM_SIZE_2) {
    size_sort_f = ceil((size_sort_f+k_nn)/2.0);
    num_batches *=2;
  }
  int size_sort = size_sort_f;

  int rowId_leaf = k_nn * blockInd + row_l;
  int g_rowId_I = leaf_id_g * ppl + rowId_leaf;
  
  //changed 
  int perm_i = g_rowId_I;

  int ind0_i = R[perm_i];
  int ind1_i = R[perm_i+1];

  float norm_i = Norms[perm_i];
  int nnz_i = ind1_i - ind0_i;
  
  //if (rowId_leaf == 0 && j == 0) printf("size_part = %d , size_sort = %d , blockDim.x = %d , num_batches = %d \n", size_part, size_sort, blockDim.x, num_batches);

  for (int n_i = j; n_i< nnz_i; n_i += blockDim.x) SM[n_i] = C[ind0_i + n_i];
   
  //int num_batches = size_part / (size_sort);
  __syncthreads();

  for (int init_write = j; init_write < SM_SIZE_2; init_write += blockDim.x) SM_dist[init_write] = 1e30;
 
  for (int col_batch = 0; col_batch < num_batches; col_batch++){
    

    for (int j_tmp = j; j_tmp < size_sort; j_tmp += blockDim.x){

      int colId_leaf = k_nn * (blockInd) + col_batch * (size_sort-k_nn) + j_tmp;
      
      if (col_batch == 0 && j_tmp < k_nn){
        
        int ind_pt = G_Id[leaf_id_g * ppl + rowId_leaf];
        int ind_read = ind_pt * k_nn + j_tmp;
        SM_dist[j_tmp] = KNN_dist[ind_read];
        SM_Id[j_tmp] = KNN_Id[ind_read];
          
      } else if (colId_leaf < ppl && j_tmp >= k_nn){

        //if (blockInd ==0 && rowId_leaf == 0 &&  colId_leaf > 4000) printf("colId_leaf = %d \n", colId_leaf);
        //if (rowId_leaf == 4091 && j_tmp < 2 * k_nn) printf("col_batch = %d , j_tmp = %d , colId_leaf = %d \n", col_batch , j_tmp, colId_leaf);
        int g_rowId_J = leaf_id_g * ppl + colId_leaf;
      

        //changed 
        int perm_j = g_rowId_J;
        
        int ind0_j = R[perm_j];
        int ind1_j = R[perm_j+1];

        int nnz_j = ind1_j - ind0_j;

        float norm_ij = norm_i + Norms[perm_j];
        
        float c_tmp = 0.0;
        int tmp_0, tmp_1, ind_jk, k, ret, testInd;
      
        ret = 0;
        testInd = 0;
        bool neighbor=false;
        
        int ind_pt = G_Id[leaf_id_g * ppl + rowId_leaf];
        int index = G_Id[perm_j];

        
        for (int ind_check = 0; ind_check < k_nn; ind_check++){
          int ind_read = ind_pt * k_nn + ind_check;  
          if (index == KNN_Id[ind_read]){
            //SM_dist[j_tmp] = 1e30;
            //SM_Id[j_tmp] = -1;
            neighbor=true;
            break; 
          }        
        }
        
        // loop over the elements of j
        if (nnz_i >0 && nnz_j > 0 && colId_leaf < ppl && neighbor == 0){
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
         
        c_tmp = -2 * c_tmp + norm_ij;
        c_tmp = (c_tmp > 1e-8) ? sqrt(c_tmp) : 0.0;
        
        //changed
        SM_dist[j_tmp] = (colId_leaf < ppl && neighbor == 0) ? c_tmp : 1e30;
        SM_Id[j_tmp] = (colId_leaf < ppl && neighbor == 0) ? G_Id[perm_j] : -1;
        
        int size_tmp = size_part - k_nn;
        int ind_tmp = leaf_id_g * k_nn * size_tmp + row_l * size_tmp + colId_leaf - (k_nn) * (blockInd+1);
        d_knn_temp[ind_tmp] = c_tmp;
      }
      //if (rowId_leaf == 0 && col_batch ==2) printf("D[%d] = %.4f , at %d , batch = %d , col = %d \n", j_tmp, SM_dist[j_tmp], SM_Id[j_tmp], col_batch, colId_leaf);
    }

 
   
    // sort and merge

    float tmp_f;
    int tmp_i;
    int ind_sort;
    __syncthreads();
    
    for (int step = 0 ; step < steps; step++){
      //for (int j_tmp =j; j_tmp < size_sort/2; j_tmp += blockDim.x){
        int j_tmp = j; 
        ind_sort = step * 2 * blockDim.x + j_tmp;
      
        int tid = sort_arr[ind_sort];
        int ixj = sort_arr_part[ind_sort];
      
        int min_max = (1 & tid);
        int coupled_flag = (1 & ixj);
      
        tid = tid >> 1;
        ixj = ixj >> 1;
      
        if (coupled_flag == 1){
        
          ind_sort += blockDim.x;
        
          int tid_1 = sort_arr[ step * 2 * blockDim.x + j_tmp + blockDim.x];
          int ixj_1 = sort_arr_part[step * 2 * blockDim.x + j_tmp + blockDim.x];
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
    //if (rowId_leaf == 0 && col_batch ==2 && j < k_nn) printf("sorted D[%d] = %.4f , at %d , batch = %d , \n", j, SM_dist[j], SM_Id[j], col_batch);
    
  }
  __syncthreads();
 
  for (int j_tmp=j; j_tmp < k_nn; j_tmp += blockDim.x){    
    if (j_tmp < k_nn){
   
      int ind_pt = leaf_id_g * ppl + rowId_leaf;
      int ind_pt_g = G_Id[ind_pt];
      int write_ind = ind_pt_g * k_nn + j_tmp;
    
       
      KNN_dist[write_ind] = SM_dist[j_tmp];
      KNN_Id[write_ind] = SM_Id[j_tmp];
    }
  }

}


__global__ void knn_kernel_B_reduced(float* KNN, int* KNN_Id, int k_nn, int ppl, int blockInd, float* d_temp_knn, int* sort_arr, int* sort_arr_part, int steps, int* G_Id, bool init){

  __shared__ float SM_dist[SM_SIZE_1];
  __shared__ int SM_Id[SM_SIZE_1];


  int j = threadIdx.x;

  int col = blockIdx.x;
  int leaf_id_g = blockIdx.z* blockDim.y + blockIdx.y;
  int colId_leaf = (init) ? col : col + k_nn * (blockInd + 1);
  int size_part = (init) ? ppl : ppl - (blockInd + 1) * (k_nn);


  int ind_tmp = (init) ? leaf_id_g * ppl * k_nn + col * k_nn + j : leaf_id_g * k_nn * size_part + j * size_part + col;
  SM_dist[j] = d_temp_knn[ind_tmp];
  int block = col / k_nn;
  int rowId_g = (init) ? leaf_id_g * ppl + block * k_nn + j : leaf_id_g * ppl + k_nn * blockInd + j;
  SM_Id[j] = G_Id[rowId_g];

  int ind_pt_knn = leaf_id_g * ppl + colId_leaf;
  int ind_pt_knn_g = G_Id[ind_pt_knn];
 
  int ind_knn = ind_pt_knn_g * k_nn + j;
  SM_dist[j + k_nn] = KNN[ind_knn];
  SM_Id[j + k_nn] = KNN_Id[ind_knn];

  //if (blockInd == 0 && init == 0 && colId_leaf == 4091) printf("read from %d , val = %.4f , ind = %d \n", ind_tmp, SM_dist[j], SM_Id[j]);
  
  __syncthreads();

  
    /*
    int index = SM_Id[j];
    for (int ind_check = 0; ind_check < k_nn; ind_check++){
      if (index == SM_Id[ind_check + k_nn]){
        SM_dist[j] = 1e30;
        SM_Id[j] = -1;
        break;
      }
    }
    */

   


  __syncthreads();

  




  float tmp_f;
  int tmp_i;

  for (int step = 0 ; step < steps; step++){

    int ind_sort = step * 2 * blockDim.x + j;
    int tid = sort_arr[ind_sort];
    int ixj = sort_arr_part[ind_sort];
    int min_max = (1 & tid);
    int coupled_flag = (1 & ixj);

    tid = tid >> 1;
    ixj = ixj >> 1;
    if (coupled_flag == 1){
        
      ind_sort += blockDim.x;
      int tid_1 = sort_arr[ind_sort];
      int ixj_1 = sort_arr_part[ind_sort];
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

  
  //ind_knn = leaf_id_g * ppl * k_nn + colId_leaf * k_nn + j;
  KNN[ind_knn] = SM_dist[j];
  KNN_Id[ind_knn] = SM_Id[j];
  
  

}


__global__ void knn_kernel_B(float* KNN, int* KNN_Id, int k_nn, int ppl, int blockInd, float* d_temp_knn, int* G_Id, bool init){

  __shared__ float SM_dist[SM_SIZE_1];
  __shared__ int SM_Id[SM_SIZE_1];

  int tid = threadIdx.x;

  int col = blockIdx.x;
  int leaf_id_g = blockIdx.z * blockDim.y + blockIdx.y;
  
  int colId_leaf = (init) ? col : col + k_nn * (blockInd + 1);
  int size_part = (init) ? ppl : ppl - (blockInd + 1) * (k_nn);
  if (tid < k_nn){
    //changed 
    int ind_tmp = (init) ? leaf_id_g * ppl * k_nn + col * k_nn + tid : leaf_id_g * k_nn * size_part + tid * size_part + col;
    SM_dist[tid] = (colId_leaf < ppl) ? d_temp_knn[ind_tmp] : 1e30;
   
    int block = col / k_nn;
    int rowId_g = (init) ? leaf_id_g * ppl + block * k_nn + tid : leaf_id_g * ppl + k_nn * blockInd + tid;  
    SM_Id[tid] = (colId_leaf < ppl) ? G_Id[rowId_g] : -1;
  } else {

    int ind_pt_knn = leaf_id_g * ppl + colId_leaf;
    int ind_pt_knn_g = G_Id[ind_pt_knn];

    int ind_knn = ind_pt_knn_g * k_nn + tid - k_nn;
    SM_dist[tid] = (colId_leaf < ppl) ? KNN[ind_knn] : 1e30;
    SM_Id[tid] = (colId_leaf < ppl) ? KNN_Id[ind_knn] : -1;
      
  }


  
  __syncthreads();
  
  if (tid < k_nn){
    int index = SM_Id[tid];
    for (int ind_check = 0; ind_check < k_nn; ind_check++){
      if (index == SM_Id[ind_check + k_nn]){
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
  
  int ind_pt = leaf_id_g * ppl + colId_leaf;    
  int ind_pt_g = G_Id[ind_pt];
  int write_ind = ind_pt_g * k_nn + tid;
   
  if (tid < k_nn) {
    KNN[write_ind] = SM_dist[tid];
    KNN_Id[write_ind] = SM_Id[tid];
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
  checkCudaErrors(cudaDeviceSynchronize());
  
  free(arr);
  free(arr_part); 
  free(tracker); 
  
}



void FIKNN_sparse_gpu(int *R, int *C, float *V, int *G_Id, int M, int leaves, int k, float *knn, int *knn_Id, int max_nnz, int iternum){



  float dt1, dt2, dt3, dt4, dt5, dt6, dt7, dt8, dt9;
  cudaEvent_t t0;
  cudaEvent_t t1;
  cudaEvent_t t2;
  cudaEvent_t t3;
  cudaEvent_t t4;
  cudaEvent_t t5;
  cudaEvent_t t6;
  cudaEvent_t t7;
  cudaEvent_t t8;
  cudaEvent_t t9;

  checkCudaErrors(cudaEventCreate(&t0));
  checkCudaErrors(cudaEventCreate(&t1));
  checkCudaErrors(cudaEventCreate(&t2));
  checkCudaErrors(cudaEventCreate(&t3));
  checkCudaErrors(cudaEventCreate(&t4));
  checkCudaErrors(cudaEventCreate(&t5));
  checkCudaErrors(cudaEventCreate(&t6));
  checkCudaErrors(cudaEventCreate(&t7));
  checkCudaErrors(cudaEventCreate(&t8));
  checkCudaErrors(cudaEventCreate(&t9));

  checkCudaErrors(cudaEventRecord(t0, 0));



  int ppl = M/leaves;


  int num_blocks_tri = ppl / k;
  //if (num_blocks_tri * k < ppl) num_blocks_tri += 1;
  int rem_len = (num_blocks_tri * k < ppl) ? ppl - num_blocks_tri * k : 0;
 
  int C_len = R[M];
  float max_nnz_f = max_nnz;
  int max_nnz_pow2 = pow(2, ceil(log2(max_nnz_f)));


  int t_b = (ppl > SM_SIZE_1) ? SM_SIZE_1 : ppl;
  //int num_batch_norm = (ppl > SM_SIZE_1) ? ppl/SM_SIZE_1 : 1;
  //if (num_batch_norm * SM_SIZE_1 < ppl) num_batch_norm += 1;
  //float tmp = leaves / 65535;
  int num_splits = 1;
  while (leaves > num_splits * 65535) num_splits *= 2;
  //int num_splits = ceil(tmp);

  //printf("leaves = %d , tmp = %.4f , num_splits = %d \n ", leaves, tmp, num_splits);
  int batch_leaves_1 = (leaves > 64000) ? leaves / num_splits : leaves;
  int batch_leaves_2 = (leaves > 64000) ? num_splits : 1;

  int verbose = 1;

  dim3 dimBlock_norm(t_b, 1, 1);
  dim3 dimGrid_norm(1, batch_leaves_1, batch_leaves_2);

  if (verbose) printf("----------------------------- Start of sfiknn ----------------------------- \n\n");

  float *d_Norms;


  int size_tri = (k > 32) ? 32 : k;
  int blockDim_tri = size_tri * (size_tri + 1)/2;
  if (blockDim_tri > SM_SIZE_1) blockDim_tri = SM_SIZE_1;

  int size_tri_last = (rem_len > 32) ? 32 : rem_len;
  int blockDim_tri_last = size_tri_last * (size_tri_last + 1)/2;
  if (blockDim_tri_last > SM_SIZE_1) blockDim_tri_last = SM_SIZE_1;

  dim3 dimBlock_tri(blockDim_tri, 1, 1);
  dim3 dimGrid_tri(num_blocks_tri, batch_leaves_1, batch_leaves_2);
  dim3 dimBlock_tri_last(blockDim_tri_last, 1, 1);
  dim3 dimGrid_tri_last(1, batch_leaves_1, batch_leaves_2);


  dim3 dimGrid_sq(k, batch_leaves_1, batch_leaves_2);
  
  if (verbose)  printf(" number of points = %d \n", M);
  if (verbose) printf(" number of pt/leaf = %d \n", ppl);
  if (verbose) printf(" number of leaves = %d \n\n", leaves);
  if (verbose) printf(" max nnz/pt = %d \n\n", max_nnz_pow2);
  if (verbose) printf(" k = %d \n\n", k);
  if (verbose) printf(" iteration = %d \n\n", iternum);

  /*
  if (verbose) printf(" dim GridThreads IterativePart = (%d , %d, %d) \n", k, batch_leaves_1, batch_leaves_2);
  if (verbose) printf(" dim BlockThreads  Norms = (%d , %d, %d) \n", t_b, 1, 1);
  if (verbose) printf(" dim GridThreads Norms = (%d , %d, %d) \n", 1, batch_leaves_1, batch_leaves_2);
  if (verbose) printf(" dim BlockThreads Diagonal Distances = (%d , %d, %d) \n", blockDim_tri, 1, 1);
  if (verbose) printf(" dim GridThreads Diagonal Distances = (%d , %d, %d) \n", num_blocks_tri, batch_leaves_1, batch_leaves_2);
  */

  int size_v_block = 2 * k;
  dim3 dimBlock_v(size_v_block, 1, 1);
  dim3 dimBlock_v_reduced(k, 1, 1);


  int *d_arr, *d_arr_part, *d_arr_v, *d_arr_part_v;
  float SM_SIZE_2_f = SM_SIZE_2;
  int n_s = log2(SM_SIZE_2_f) *(log2(SM_SIZE_2_f)+1) /2;

  int copy_size = (ppl) * n_s;
  float tmp = 2*k;
  int n_s_v = log2(tmp) * (log2(tmp)+1)/2;
  int copy_size_v = k * n_s;

  size_t free, total, m1, m2, m3;

  int *d_R, *d_GId, *d_C, *d_knn_Id;
  float *d_V, *d_knn;  
  
  
  
  checkCudaErrors(cudaMalloc((void **) &d_R, sizeof(int) * (M+1)));
  checkCudaErrors(cudaMalloc((void **) &d_GId, sizeof(int) * M));
  checkCudaErrors(cudaMalloc((void **) &d_C, sizeof(int) * C_len));
  checkCudaErrors(cudaMalloc((void **) &d_V, sizeof(float) * C_len));

  checkCudaErrors(cudaMalloc((void **) &d_knn_Id, sizeof(int) *M*k));
  checkCudaErrors(cudaMalloc((void **) &d_knn, sizeof(float) *M*k));

  checkCudaErrors(cudaMemcpy(d_R, R, sizeof(int) * (M+1), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_C, C, sizeof(int) * C_len, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_V, V, sizeof(float) * C_len, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_GId, G_Id, sizeof(int) * M, cudaMemcpyHostToDevice)); 
  checkCudaErrors(cudaMemcpy(d_knn_Id, G_Id, sizeof(int) * M, cudaMemcpyHostToDevice)); 
  checkCudaErrors(cudaMemcpy(d_knn, knn, sizeof(float) * M * k, cudaMemcpyHostToDevice)); 
  checkCudaErrors(cudaMemcpy(d_knn_Id, knn_Id, sizeof(int) * M * k, cudaMemcpyHostToDevice)); 

  cudaMemGetInfo(&free, &total);
  checkCudaErrors(cudaMalloc((void **) &d_arr, sizeof(int) * copy_size));
  checkCudaErrors(cudaMalloc((void **) &d_arr_part, sizeof(int) * copy_size));

  checkCudaErrors(cudaMalloc((void **) &d_arr_v, sizeof(int) * copy_size_v));
  checkCudaErrors(cudaMalloc((void **) &d_arr_part_v, sizeof(int) * copy_size_v));


  checkCudaErrors(cudaMemset(d_arr, 0, sizeof(int) * copy_size));
  checkCudaErrors(cudaMemset(d_arr_part, 0, sizeof(int) * copy_size));
  checkCudaErrors(cudaMemset(d_arr_v, 0, sizeof(int) * copy_size_v));
  checkCudaErrors(cudaMemset(d_arr_part_v, 0, sizeof(int) * copy_size_v));
  cudaMemGetInfo(&m1, &total);



  checkCudaErrors(cudaEventRecord(t1, 0));


  precomp_arbsize_sortId(d_arr_v, d_arr_part_v, 2*k, 2*k, n_s_v, copy_size_v);


  checkCudaErrors(cudaEventRecord(t2, 0));

  float * d_temp_knn;
  checkCudaErrors(cudaMalloc((void **) &d_Norms, sizeof(float) * M));

  cudaMemGetInfo(&m2, &total);
  checkCudaErrors(cudaMalloc((void **) &d_temp_knn, sizeof(float) * M * k));
  cudaMemGetInfo(&m3, &total);

  int steps;



  FIKNN_compute_norm <<< dimGrid_norm, dimBlock_norm >>>(d_R, d_C, d_V, d_GId, d_Norms, ppl);
  checkCudaErrors(cudaDeviceSynchronize());

  checkCudaErrors(cudaEventRecord(t3, 0));
  
  knn_kernel_tri <<< dimGrid_tri, dimBlock_tri >>>(d_R, d_C, d_V, d_GId, d_Norms, k, d_temp_knn, ppl, max_nnz_pow2, iternum);
  checkCudaErrors(cudaDeviceSynchronize());
  
  if (rem_len > 0) {
    knn_kernel_tri_last <<< dimGrid_tri_last, dimBlock_tri_last >>>(d_R, d_C, d_V, d_GId, d_Norms, k, d_temp_knn, ppl, max_nnz_pow2, iternum, rem_len, num_blocks_tri);
    checkCudaErrors(cudaDeviceSynchronize());
  }


  int size_v = ppl;
  //dim3 dimGrid_v(size_v, batch_leaves_1, batch_leaves_2);
  dim3 dimGrid_v2(size_v, batch_leaves_1, batch_leaves_2);
    
  knn_kernel_B_reduced <<< dimGrid_v2, dimBlock_v_reduced >>> (d_knn, d_knn_Id, k, ppl, 0, d_temp_knn, d_arr_v, d_arr_part_v, n_s_v, d_GId, true);
  //knn_kernel_B <<< dimGrid_v, dimBlock_v >>> (d_knn, d_knn_Id, k, ppl, 0, d_temp_knn, d_GId, true);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaEventRecord(t4, 0));
  
  int num_iters = (rem_len > 0) ? num_blocks_tri : num_blocks_tri - 1;
  for (int blockInd = 0; blockInd < num_iters; blockInd++){

    //checkCudaErrors(cudaEventRecord(t6, 0));
    
    int size_part = ppl - blockInd *k;
    int size_sort = size_part;

    while (size_sort > SM_SIZE_2) size_sort = ceil((size_sort+k)/2);
    float tmp = size_sort/2.0;
    int blocksize = ceil(tmp);
    float tmp_f = 2 * blocksize;
    int N_pow2 = pow(2, ceil(log2(tmp_f)));
    tmp_f = N_pow2;
    steps = log2(tmp_f) * (log2(tmp_f) +1)/2;
    //if (blockInd == 0) printf("size_part = %d , size_sort = %d , blockSize = %d , N_pow2 = %d , steps = %d , copy_size = %d  \n", size_part, size_sort, blocksize, N_pow2, steps, copy_size); 
    


    int real_size = 2 * blocksize;
    precomp_arbsize_sortId(d_arr, d_arr_part, real_size, N_pow2, steps, copy_size);
    //checkCudaErrors(cudaEventRecord(t9, 0));


    dim3 dimBlock_sq(blocksize, 1, 1);



    int size_v2 = ppl - (blockInd + 1) * k;
    dim3 dimGrid_v2(size_v2, batch_leaves_1, batch_leaves_2);

    knn_kernel_A <<< dimGrid_sq, dimBlock_sq >>>(d_R, d_C, d_V, d_GId, d_Norms, k, d_knn, d_knn_Id, ppl, max_nnz_pow2, blockInd, d_arr, d_arr_part, steps, d_temp_knn);
    checkCudaErrors(cudaDeviceSynchronize());
    //checkCudaErrors(cudaEventRecord(t7, 0));
    //knn_kernel_B <<< dimGrid_v2, dimBlock_v >>> (d_knn, d_knn_Id, k, ppl, blockInd, d_temp_knn, d_GId, false);
    knn_kernel_B_reduced <<< dimGrid_v2, dimBlock_v_reduced >>> (d_knn, d_knn_Id, k, ppl, blockInd, d_temp_knn, d_arr_v, d_arr_part_v, n_s_v, d_GId, false);

    checkCudaErrors(cudaDeviceSynchronize());
    //checkCudaErrors(cudaEventRecord(t8, 0));
    //checkCudaErrors(cudaEventElapsedTime(&dt7, t6, t7));
    //checkCudaErrors(cudaEventElapsedTime(&dt8, t7, t8));
    //checkCudaErrors(cudaEventElapsedTime(&dt9, t6, t9));
    //printf("itera = %d , horiz = %.4f (%.4f) , , vertical = %.4f (%.4f) , precomp = %.4f \n", blockInd, dt7/1000, dt7/(dt7+dt8), dt8/1000, dt8/(dt7+dt8), dt9/1000); 
  }
  


  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaEventRecord(t5, 0));
  checkCudaErrors(cudaEventSynchronize(t5));
  checkCudaErrors(cudaEventElapsedTime(&dt1, t0, t1));
  checkCudaErrors(cudaEventElapsedTime(&dt2, t1, t2));
  checkCudaErrors(cudaEventElapsedTime(&dt3, t2, t3));
  checkCudaErrors(cudaEventElapsedTime(&dt4, t3, t4));
  checkCudaErrors(cudaEventElapsedTime(&dt5, t4, t5));
  checkCudaErrors(cudaEventElapsedTime(&dt6, t0, t5));

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
  //cudaMemGetInfo(&free, &total);
  if (verbose) printf("Timings \n");
  if (verbose) printf("Memory allocation = %.4f (%.4f %%) \n", dt1/1000, dt1/dt6);
  if (verbose) printf("Precomp sortId (vertical)  = %.4f (%.4f %%) \n", dt2/1000, dt2/dt6);
  if (verbose) printf("Computing norms = %.4f (%.4f %%) \n", dt3/1000, dt3/dt6);
  if (verbose) printf("Diagonal part = %.4f (%.4f %%) \n", dt4/1000, dt4/dt6);
  if (verbose) printf("Iterative part = %.4f (%.4f %%) \n", dt5/1000, dt5/dt6);
  if (verbose) printf("Total = %.4f \n", dt6/1000);

  if (verbose) printf(" Memory : storing norms = %.4f GB \n", (m1-m2)/1e9);
  if (verbose) printf(" Memory : precomputing the sort indices = %.4f GB \n", (free-m1)/1e9);
	if (verbose) printf(" Memory : temporary storage = %.4f GB \n", (m2-m3)/1e9);
  
  if (verbose) printf("----------------------------- End of the sfiknn -----------------------------\n\n");

}











