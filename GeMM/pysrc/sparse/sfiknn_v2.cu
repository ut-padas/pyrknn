
#define SM_SIZE_1 1024
#define SM_SIZE_2 2048
#define SM_SIZE_SORT 8192

#include "sfiknn.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>


__global__ void ComputeNorms(int* R, int* C, float* V, int* G_Id, float* Norms, int ppl) {

  int ind = threadIdx.x;
  int leaf_id_g = blockIdx.z * blockDim.y + blockIdx.y;
  for (int row = ind; row < ppl; row += blockDim.x){
    int g_rowId = leaf_id_g * ppl + row;
    //changed
  
    int g_Id = g_rowId;

   
    int ind0_i = R[g_Id];
 
    int nnz = R[g_Id + 1] - ind0_i;
    float norm_i = 0.0;
    float v;
    for (int n_i = 0; n_i < nnz; n_i += 1) {
      v = V[ind0_i + n_i];
      norm_i += v * v;
      //if (G_Id[g_rowId] == 777478) printf("norm = %.4f, V[0] = %.4f , R = %d , nnz = %d \n", norm_i, v, ind0_i, nnz);
    }
    Norms[g_Id] = norm_i;
  }
}

__global__ void ComputeTriDists(int* R, int* C, float* V, int* G_Id, float* Norms , int k_nn, float* KNN_dist_tmp, int ppl, int bl, int sizebleaves, int partsize) {



  int ind = threadIdx.x;
  int leafId_local = blockIdx.z * blockDim.y + blockIdx.y;
  int leafId_g = bl * sizebleaves + leafId_local;
;
  int block = blockIdx.x;



  int size_block = partsize * (partsize + 1) /2;
  
  for (int elem = ind; elem < size_block; elem += blockDim.x){

    float tmp = -8 * elem + 4 * partsize * (partsize+1) - 7;
    int rowId = sqrt(tmp)/2.0 - 0.5;
    rowId = partsize - 1 - rowId;
    int colId = elem + rowId - partsize * (partsize + 1) / 2 + (partsize - rowId) * ((partsize - rowId) + 1)/2;

    float c_tmp = 0.0;

    int g_rowId = leafId_g * ppl + block * partsize + rowId;
    int g_colId = leafId_g * ppl + block * partsize + colId;
   
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
    //if (G_Id[leafId_g * ppl + block * partsize + rowId] == 777478) printf("D[%d] = %.4f for %d , leaf = %d \n", colId , c_tmp, G_Id[leafId_g * ppl + block * partsize + colId], leafId_local); 
    //if (G_Id[leafId_g * ppl + block * partsize + colId] == 777478) printf("D[%d] = %.4f for %d , leaf = %d \n", rowId , c_tmp, G_Id[leafId_g * ppl + block * partsize + rowId], leafId_local); 
    
    // changed 
    int gid_pt = leafId_local * ppl + block * partsize + rowId;
    int gid_pt_T = leafId_local * ppl + block * partsize + colId;
    int ind_knn = gid_pt * partsize + colId;
    int ind_knn_T = gid_pt_T * partsize + rowId;
    //KNN_dist_tmp[ind_knn] = (iternum >0 && colId == rowId) ? 1e30 : c_tmp;
    
    //if (leafId_g ==0 && g_rowId == 0) printf("D[%d] = %.4f write at %d \n", colId, c_tmp, ind_knn);
    //if (ind_knn < 0 || ind_knn_T < 0 || ind_knn >= gridDim.y * ppl * partsize || ind_knn_T >= gridDim.y * ppl * partsize) printf("print err ind_knn = %d , ind_knn_T = %d \n", ind_knn, ind_knn_T);
    
    KNN_dist_tmp[ind_knn] = c_tmp;
    if (colId > rowId) KNN_dist_tmp[ind_knn_T] = c_tmp;
    
  }
  
}



__global__ void ComputeTriDists_last(int* R, int* C, float* V, int* G_Id, float* Norms , int k_nn, float* KNN_dist_tmp, int ppl, int rem_len , int blockId, int bl, int sizebleaves, int partsize) {




  int ind = threadIdx.x;
  int leafId_local = blockIdx.z * blockDim.y + blockIdx.y;
  int leafId_g = bl * sizebleaves + leafId_local;
  int block = blockId;
  


  int size_block = rem_len * (rem_len + 1) /2;


  for (int elem = ind; elem < size_block; elem += blockDim.x){

    float tmp = -8 * elem + 4 * rem_len * (rem_len+1) - 7;
    int rowId = sqrt(tmp)/2.0 - 0.5;
    rowId = rem_len - 1 - rowId;
    int colId = elem + rowId - rem_len * (rem_len + 1) / 2 + (rem_len - rowId) * ((rem_len - rowId) + 1)/2;

    float c_tmp = 0.0;
    if (block * partsize + rowId < ppl && block * partsize + colId < ppl){

    int g_rowId = leafId_g * ppl + block * partsize + rowId;
    int g_colId = leafId_g * ppl + block * partsize + colId;

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
    int gid_pt = leafId_local * ppl + block * partsize + rowId;
    int gid_pt_T = leafId_local * ppl + block * partsize + colId;
    int ind_knn = gid_pt * partsize + colId;
    int ind_knn_T = gid_pt_T * partsize + rowId;

    //KNN_dist_tmp[ind_knn] = (iternum >0 && colId == rowId) ? 1e30 : c_tmp;
    //if (G_Id[leafId_g * ppl + block * partsize + rowId] == 777478) printf("D[%d] = %.4f for %d , leaf = %d \n", colId , c_tmp, G_Id[leafId_g * ppl + block * partsize + colId], leafId_local); 
    //if (G_Id[leafId_g * ppl + block * partsize + colId] == 777478) printf("T D[%d] = %.4f for %d , leaf = %d \n", rowId , c_tmp, G_Id[leafId_g * ppl + block * partsize + rowId], leafId_local); 
    KNN_dist_tmp[ind_knn] = c_tmp;
    if (colId > rowId) KNN_dist_tmp[ind_knn_T] = c_tmp;
    /*
    for (int row_tmp = 0; row_tmp<rem_len; row_tmp++){
      for (int q = ind + rem_len; q < partsize; q += blockDim.x){
        gid_pt = leafId_local * ppl + block * partsize + row_tmp;
        ind_knn = gid_pt * partsize + q;
        if (ind_knn < leafId_local * ppl * partsize) KNN_dist_tmp[ind_knn] = 1e30;
      } 
    }
    */
    /*
    int max_write = gridDim.y * gridDim.z * ppl * partsize;
    int size_pad = (partsize - rem_len - 1) * rem_len;
    for (int elem = ind; elem < size_pad; elem += blockDim.x){
      int row_tmp = size_pad / (partsize - rem_len - 1);
      int col_tmp = size_pad - row_tmp * (partsize - rem_len - 1) + rem_len;
      //row_tmp += rem_len;
      gid_pt = leafId_local * ppl * partsize + block * partsize * partsize + row_tmp * partsize;
      ind_knn = gid_pt + col_tmp;
      if (G_Id[leafId_g * ppl + block * partsize + row_tmp] == 777478) printf("padding (%d, %d), write at %d , max = %d \n", row_tmp, col_tmp, ind_knn, max_write);
      if (ind_knn >= max_write) printf("leaf = %d , row_tmp = %d , col_tmp = %d , block = %d \n", leafId_local, row_tmp, col_tmp, block);
      KNN_dist_tmp[ind_knn] = 1e30;  
    }
    */

  }

}

__global__ void ComputeRecDists(int* R, int* C, float* V, int* G_Id, float* Norms, int k_nn, int ppl, int blockInd, float* d_knn_temp, int bl, int sizebleaves, int partsize) {


  __shared__ int SM[1000];
  
  int row_l = blockIdx.x;
  int leafId_local = blockIdx.z * blockDim.y + blockIdx.y;
  int leafId_g = bl * sizebleaves + leafId_local;
  int j = threadIdx.x;
  
  int size_part = ppl - (partsize) * (blockInd+1); 
  


  int rowId_leaf = partsize * blockInd + row_l;
  int g_rowId_I = leafId_g * ppl + rowId_leaf;
  
  //changed 
  int perm_i = g_rowId_I;

  int ind0_i = R[perm_i];
  int ind1_i = R[perm_i+1];

  float norm_i = Norms[perm_i];
  int nnz_i = ind1_i - ind0_i;
  

  for (int n_i = j; n_i< nnz_i; n_i += blockDim.x) SM[n_i] = C[ind0_i + n_i];
   
  __syncthreads();


  for (int j_tmp = j; j_tmp < size_part; j_tmp += blockDim.x){

    int colId_leaf = partsize * (blockInd+1) + j_tmp;
      
    int g_rowId_J = leafId_g * ppl + colId_leaf;
      
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
      
    // loop over the elements of j
  
    if (nnz_i >0 && nnz_j > 0 && colId_leaf < ppl){
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
        
        
    int size_tmp = size_part;
    int ind_tmp = leafId_local * partsize * size_tmp + row_l * size_tmp + colId_leaf - (partsize) * (blockInd+1);
    d_knn_temp[ind_tmp] = c_tmp;
    //if (G_Id[leafId_local * ppl + rowId_leaf] == 777478) printf("D[%d] = %.4f for %d , leaf = %d \n", j_tmp , c_tmp, G_Id[leafId_local * ppl + colId_leaf], leafId_local); 
  }


}
 



__global__ void MergeHoriz(float* KNN, int* KNN_Id, int k_nn, int ppl, int blockInd, float* d_temp_knn, int* sort_arr, int* sort_arr_part, int steps, int* G_Id, bool init, int bl, int sizebleaves, int partsize){

   
  __shared__ float SM_dist[SM_SIZE_2];
  __shared__ int SM_Id[SM_SIZE_2];



  int j = threadIdx.x;
  int row_l = blockIdx.x;
  int leafId_local = blockIdx.z * blockDim.y + blockIdx.y;
  int leafId_g = bl * sizebleaves + leafId_local; 

  int size_part = ppl - (partsize) * blockInd;
  int size_sort = 2 * blockDim.x;

  int rowId_leaf = partsize * blockInd + row_l;
  //int g_rowId_I = leaf_id_g * ppl + rowId_leaf;
  
  for (int n=j; n < SM_SIZE_2; n += blockDim.x){
    SM_dist[n] = 1e30; 
    SM_Id[n] = -1;
  }

  float tmp_f;
  int tmp_i;
  int ind_sort;
   
  int num_batches = size_part / (size_sort - k_nn);
  if (size_part > 0 && num_batches == 0) num_batches += 1;

  
  for (int col_batch = 0; col_batch < num_batches; col_batch++){
    for (int j_tmp = j; j_tmp < size_sort; j_tmp += blockDim.x){
      
      int colId_leaf = (col_batch == 0) ? partsize * (blockInd + 1) + j_tmp - k_nn : partsize * blockInd + col_batch * (size_sort - k_nn) + j_tmp;
      
      if (col_batch == 0 && j_tmp < k_nn){
        
        int ind_pt = G_Id[leafId_g * ppl + rowId_leaf];
        int ind_read = ind_pt * k_nn + j_tmp;
        SM_dist[j_tmp] = KNN[ind_read];
        SM_Id[j_tmp] = KNN_Id[ind_read];
        //if (ind_pt ==  777478) printf(" Neighbors , D[%d] = %.4f at %d \n", j_tmp, SM_dist[j_tmp], SM_Id[j_tmp]);
      } else if (colId_leaf < ppl && j_tmp >= k_nn){

        int size_tmp = size_part - partsize;
        //int ind_tmp = leafId_local * partsize * size_tmp + row_l * size_tmp + colId_leaf - (k_nn) * (blockInd+1);
        int ind_tmp = leafId_local * partsize * size_tmp + row_l * size_tmp + colId_leaf - (partsize) * (blockInd+1);
        int g_colId_J = leafId_g * ppl + colId_leaf;
        
        SM_dist[j_tmp] = d_temp_knn[ind_tmp];
        SM_Id[j_tmp] = G_Id[g_colId_J];
        //if (G_Id[leafId_g * ppl + rowId_leaf] ==  777478) printf(" Dists tmp , D[%d] = %.4f at %d \n", j_tmp, SM_dist[j_tmp], SM_Id[j_tmp]);
        //if (leafId_g == 0 && rowId_leaf == 782) printf(" Dists , D[%d] = %.4f at %d \n", j_tmp, SM_dist[j_tmp], SM_Id[j_tmp]);
      }
      //if (init == 0 && rowId_leaf == 1149 && leaf_id_g == 0 && col_batch == 0) printf("D[%d] = %.4f , at %d  \n", j_tmp, SM_dist[j_tmp], SM_Id[j_tmp]);
    }

    __syncthreads();
       
    for (int j_tmp = j; j_tmp < size_sort; j_tmp += blockDim.x) {

      if (j_tmp >= k_nn){
        int index = SM_Id[j_tmp];
        for (int ind_check = 0; ind_check < k_nn; ind_check++){
          if (index == SM_Id[ind_check]){
            SM_Id[j_tmp] = -1;
            SM_dist[j_tmp] = 1e30;
            break;
          }
        }
      }

    }
    __syncthreads();
    


    for (int step = 0; step < steps; step++){
    
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


  }
  for (int j_tmp = j; j_tmp < k_nn; j_tmp += blockDim.x){ 
    if (j_tmp < k_nn){
      int ind_pt = leafId_g * ppl + rowId_leaf;
      int write_ind = G_Id[ind_pt] * k_nn + j_tmp;
      KNN[write_ind] = SM_dist[j_tmp];
      KNN_Id[write_ind] = SM_Id[j_tmp];
      //if (ind_pt ==  777478) printf(" Sorted Horiz Dists  , D[%d] = %.4f at %d \n", j_tmp, SM_dist[j_tmp], SM_Id[j_tmp]);
      //if (leafId_g == 0 && rowId_leaf == 782) printf("Block = %d,  Sorted Neighbors , D[%d] = %.4f at %d , write at %d \n", blockInd, j_tmp, SM_dist[j_tmp], SM_Id[j_tmp], write_ind);
      //if (init == 0 && rowId_leaf == 1149 && leaf_id_g == 0) printf("sorted D[%d] = %.4f , at %d  \n", j_tmp, SM_dist[j_tmp], SM_Id[j_tmp]);
    }
  } 
}


__global__ void MergeVer_v2(float* KNN, int* KNN_Id, int k_nn, int ppl, int blockInd, float* d_temp_knn, int* G_Id, bool init, int bl, int sizebleaves, int partsize){

  __shared__ float SM_dist[SM_SIZE_1];
  __shared__ int SM_Id[SM_SIZE_1];


  int j = threadIdx.x;
  float tmp_f;
  int tmp_i;

  int col = blockIdx.x;
  int leafId_local = blockIdx.z* blockDim.y + blockIdx.y;
  int leafId_g = bl * sizebleaves + leafId_local;

  int colId_leaf = (init) ? col : col + partsize * (blockInd + 1);
  int size_part = (init) ? ppl : ppl - (blockInd + 1) * partsize;


  //float norm_i = Norms[leafId_g * ppl + colId_leaf];

  int ind_pt_knn = leafId_g * ppl + colId_leaf;
  int ind_pt_knn_g = G_Id[ind_pt_knn];

  for (int ind = j; ind < 2 * partsize; ind += blockDim.x){
    SM_dist[ind] = 1e30;
    SM_Id[ind] = -1;
  }
  __syncthreads();


  for (int j_tmp = j; j_tmp < partsize; j_tmp += blockDim.x){

    //int ind_tmp = (init) ? leafId_local * ppl * partsize + col * partsize + j_tmp : leafId_local * partsize * size_part + j_tmp + col * partsize;
    int ind_tmp = (init) ? leafId_local * ppl * partsize + col * partsize + j_tmp : leafId_local * partsize * size_part + j_tmp * size_part + col;

    int block = col / partsize;
    int M = ppl * gridDim.y * gridDim.z;
    
    int rowId_g = (init) ? leafId_g * ppl + block * partsize + j_tmp : leafId_g * ppl + partsize * blockInd + j_tmp;

    int Max_blocks = ppl / partsize;
    int rem_len = ppl - Max_blocks * partsize;

    SM_Id[j_tmp] = (rowId_g < M) ? G_Id[rowId_g] : -1;
    SM_dist[j_tmp] = (Max_blocks == block && j_tmp >= rem_len && init == 1) ? 1e30 : d_temp_knn[ind_tmp];
    //tmp_f = (rowId_g < M) ? -2 * d_temp_knn[ind_tmp] + norm_i + Norms[rowId_g] : 1e30;
    //tmp_f = d_temp_knn[ind_tmp] + norm_i + Norms[rowId_g];

    /*
    if (init){
      tmp_f = (block * partsize + j_tmp < ppl) ? tmp_f : 1e30;
    }

    SM_dist[j_tmp] = (tmp_f > 0.0) ? sqrt(tmp_f) : 0.0;
    */
    //if (ind_pt_knn_g == 777478 && blockInd == 0 && init == 1) printf("Ver read tmp D[%d] = %.4f at %d \n", j_tmp, SM_dist[j_tmp], SM_Id[j_tmp]);

    int ind_knn = ind_pt_knn_g * k_nn + j_tmp;
    if (j_tmp < k_nn){
      SM_dist[j_tmp + partsize] = KNN[ind_knn];
      SM_Id[j_tmp + partsize] = KNN_Id[ind_knn];
      //if (ind_pt_knn_g == 777478 && blockInd == 0 && init == 1) printf("Ver read Nhbd D[%d] = %.4f at %d \n", j_tmp + partsize, SM_dist[j_tmp + partsize], SM_Id[j_tmp + partsize]);
     
    } else {
      SM_dist[j_tmp + partsize] = 1e30;
      SM_Id[j_tmp + partsize] = -1;
    }


  }

  __syncthreads();


  for (int j_tmp = j; j_tmp < partsize; j_tmp += blockDim.x){
    int index = SM_Id[j_tmp];
    for (int ind_check = 0; ind_check < k_nn; ind_check++){
      if (index == SM_Id[ind_check + partsize]){
        SM_dist[j_tmp] = 1e30;
        SM_Id[j_tmp] = -1;
        break;
      }
    }
  }
  __syncthreads();



  //int size_sort = 2 * partsize;


  for (int g = 2; g <= 2 * blockDim.x; g *= 2){
    for (int l = g/2; l > 0; l /= 2){
      int tid = j;
      int ixj = tid ^ l;
      if (ixj < tid){
        tid += blockDim.x;
        ixj = tid ^ l;
        if (ixj < tid){
          tid = ixj;
          ixj = tid ^ l;
        }
      }

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
      __syncthreads();
    }
  }

  if (j < k_nn){
    int ind_pt_knn = leafId_g * ppl + colId_leaf;
    int ind_pt_knn_g = G_Id[ind_pt_knn];
    int ind_knn = ind_pt_knn_g * k_nn + j;
    //if (ind_pt_knn_g == 777478 && blockInd == 0) printf("Sorted Ver D[%d] = %.4f at %d \n", j, SM_dist[j], SM_Id[j]);
    KNN[ind_knn] = SM_dist[j];
    KNN_Id[ind_knn] = SM_Id[j];
    //if (leafId_g == 0 && colId_leaf == 782) printf("BlockInd = %d , Sorted Ver D[%d] = %.4f , at %d , write at %d \n", blockInd, j, SM_dist[j], SM_Id[j], ind_knn); 
  }


}


void PrecompSortIdsSparse(int* d_arr, int* d_arr_part, int N_true, int N_pow2, int steps, int copy_size){

  
  
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



void sfi_leafknn(int *d_R, int *d_C, float *d_V, int *d_GId, int M, int leaves, int k, float *d_knn, int *d_knn_Id){


  printf("----------------------------- Starting SpIF knn ----------------------------- \n\n");

  float dt1, dt2, dt3, dt4, dt5, dt6, dt7, dt8, dt9, dt_tmp;
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

  int verbose = 1;
	if (verbose) printf("----------------------------- Start of sfiknn ----------------------------- \n\n");

  size_t free, total, m1, m2, m3;

  int ppl = M/leaves;

  int partsize = (k > 32) ? k : 32;
  
  cudaMemGetInfo(&free, &total);
  printf(" Available Memory : %.4f GB from %.4f \n", free/1e9, total/1e9);
  

  size_t size_req = sizeof(float) * partsize * M;
  int counter=0;
	while (size_req < free && partsize < 256 && counter < 6) {
    counter++;
    size_req *= 2;
    partsize *= 2;
    //printf("partsize = %d,  free = %.4f , size_req = %.4f,\n", partsize, free/1e9, size_req/1e9);
  }
  partsize /= 2;
  partsize = (partsize > ppl) ? ppl : partsize;
  partsize = (partsize < k) ? k : partsize;  


  int num_blocks_tri = ppl / partsize;
  int rem_len = (num_blocks_tri * partsize < ppl) ? ppl - num_blocks_tri * partsize : 0;
 

  int t_b = (ppl > SM_SIZE_1) ? SM_SIZE_1 : ppl;

  int num_splits = 1;
  while (leaves > num_splits * 65535) num_splits *= 2;

  int batch_leaves_1 = (leaves > 64000) ? leaves / num_splits : leaves;
  int batch_leaves_2 = (leaves > 64000) ? num_splits : 1;

  


  float *d_Norms;


  //int size_tri = (k > 32) ? 32 : k;
  int size_tri = partsize;
 
  int blockDim_tri = size_tri * (size_tri + 1)/2;
  if (blockDim_tri > SM_SIZE_1) blockDim_tri = SM_SIZE_1;

  int size_tri_last = (rem_len > 32) ? 32 : rem_len;
  int blockDim_tri_last = size_tri_last * (size_tri_last + 1)/2;
  if (blockDim_tri_last > SM_SIZE_1) blockDim_tri_last = SM_SIZE_1;

  dim3 BlockDistTri(blockDim_tri, 1, 1);
  dim3 GridDistTri(num_blocks_tri, batch_leaves_1, batch_leaves_2);

  dim3 BlockDistTri_last(blockDim_tri_last, 1, 1);
  dim3 GridDistTri_last(1, batch_leaves_1, batch_leaves_2);
  
  dim3 BlockNorm(t_b, 1, 1);
  dim3 GridNorm(1, batch_leaves_1, batch_leaves_2);

  dim3 GridDistRec(partsize, batch_leaves_1, batch_leaves_2);
  
  dim3 BlockMergeVer(partsize, 1, 1);
  
  dim3 GridMergeHoriz(partsize, batch_leaves_1, batch_leaves_2);


  printf("=======================\n");
  printf(" Num points = %d \n", M);
  printf(" pt/leaf = %d \n", ppl);
  printf(" Leaves = %d \n", leaves);
  printf(" K = %d \n", k);
  printf(" PartSize = %d \n", partsize);

  printf(" BlockThreads Norms = (%d , %d, %d) \n", BlockNorm.x, BlockNorm.y, BlockNorm.z);
  printf(" dim GridThreads Norms = (%d , %d, %d) \n", GridNorm.x, GridNorm.y, GridNorm.z);
  printf(" dim GridThreads MergeHoriz = (%d , %d, %d) \n", GridMergeHoriz.x, GridMergeHoriz.y, GridMergeHoriz.z);
  printf(" dim BlockThreads MergeVer = (%d , %d, %d) \n", BlockMergeVer.x, BlockMergeVer.y, BlockMergeVer.z);
  printf(" dim BlockThreads Diagonal Distances = (%d , %d, %d) \n", BlockDistTri.x, BlockDistTri.y, BlockDistTri.z);
  printf(" dim GridThreads Diagonal Distances = (%d , %d, %d) \n", GridDistTri.x, GridDistTri.y, GridDistTri.z);
  printf(" dim GridThreads RecDistances = (%d, %d, %d) \n", GridDistRec.x, GridDistRec.y, GridDistRec.z);

  /*
  if (verbose) printf(" dim GridThreads IterativePart = (%d , %d, %d) \n", k, batch_leaves_1, batch_leaves_2);
  if (verbose) printf(" dim BlockThreads  Norms = (%d , %d, %d) \n", t_b, 1, 1);
  if (verbose) printf(" dim GridThreads Norms = (%d , %d, %d) \n", 1, batch_leaves_1, batch_leaves_2);
  */



  int *d_arr, *d_arr_part;
  float SM_SIZE_2_f = SM_SIZE_2;
  int n_s = log2(SM_SIZE_2_f) *(log2(SM_SIZE_2_f)+1) /2;

  int copy_size = (ppl) * n_s;
  float tmp = 2*partsize;
  int n_s_v = log2(tmp) * (log2(tmp)+1)/2;

  //size_t free, total, m1, m2, m3;

  //int *d_R, *d_GId, *d_C, *d_knn_Id;
  //float *d_V, *d_knn;  
    

  cudaMemGetInfo(&free, &total);
  checkCudaErrors(cudaMalloc((void **) &d_arr, sizeof(int) * copy_size));
  checkCudaErrors(cudaMalloc((void **) &d_arr_part, sizeof(int) * copy_size));

  checkCudaErrors(cudaMemset(d_arr, 0, sizeof(int) * copy_size));
  checkCudaErrors(cudaMemset(d_arr_part, 0, sizeof(int) * copy_size));
  cudaMemGetInfo(&m1, &total);

  checkCudaErrors(cudaEventRecord(t1, 0));
  checkCudaErrors(cudaMalloc((void **) &d_Norms, sizeof(float) * M));

  cudaMemGetInfo(&m2, &total);

  size_t size_tmp = sizeof(float) * M * partsize;
  printf(" Needed %.4f GB , avail : %.4f \n", size_tmp/1e9, m2/1e9);
  float tmp2 = ceil(size_tmp/m2);
  
  int bleaves = (size_tmp > m2) ? log2(tmp2) : 0;
  int numbleaves = 1 << bleaves;
  int sizebleaves = leaves / numbleaves;

  printf(" Num BatchLeaves = %d \n", numbleaves);
  printf(" Size BatchLeaves = %d \n", sizebleaves);
  printf("======================\n");


  float * d_temp_knn;
  checkCudaErrors(cudaMalloc((void **) &d_temp_knn, sizeof(float) * M * partsize));
  cudaMemGetInfo(&m3, &total);



  checkCudaErrors(cudaEventRecord(t2, 0));


  int steps;

  ComputeNorms <<< GridNorm, BlockNorm >>>(d_R, d_C, d_V, d_GId, d_Norms, ppl);
  checkCudaErrors(cudaDeviceSynchronize());

  checkCudaErrors(cudaEventRecord(t3, 0));

  for (int bl = 0; bl < numbleaves; bl++){

  
    ComputeTriDists <<< GridDistTri, BlockDistTri >>>(d_R, d_C, d_V, d_GId, d_Norms, k, d_temp_knn, ppl, bl, sizebleaves, partsize);
    checkCudaErrors(cudaDeviceSynchronize());
  
    if (rem_len > 0) {
      ComputeTriDists_last <<< GridDistTri_last, BlockDistTri_last >>>(d_R, d_C, d_V, d_GId, d_Norms, k, d_temp_knn, ppl,  rem_len, num_blocks_tri, bl, sizebleaves, partsize);
      checkCudaErrors(cudaDeviceSynchronize());
    }

    int size_v = ppl;
    dim3 GridMergeVer(size_v, batch_leaves_1, batch_leaves_2);    
		//MergeVer_v2 <<< GridMergeVer, BlockMergeVer >>> (d_knn, d_knn_Id, k, ppl, 0, d_temp_knn, d_arr_v, d_arr_part_v, n_s_v, d_GId, true, M);
		MergeVer_v2 <<< GridMergeVer, BlockMergeVer >>> (d_knn, d_knn_Id, k, ppl, 0, d_temp_knn, d_GId, true, bl, sizebleaves, partsize );
		checkCudaErrors(cudaDeviceSynchronize());
		checkCudaErrors(cudaEventRecord(t4, 0));
  
		int num_iters = (rem_len > 0) ? num_blocks_tri : num_blocks_tri - 1;
 
		dt5 = 0.0;
		dt6 = 0.0;
		dt7 = 0.0;
 
		for (int blockInd = 0; blockInd < num_iters; blockInd++){

			checkCudaErrors(cudaEventRecord(t5, 0));
			checkCudaErrors(cudaEventSynchronize(t5));
			
			int size_part = ppl - blockInd *k;
			int size_sort = size_part;

			while (size_sort > SM_SIZE_2) size_sort = ceil((size_sort+k)/2);
		 
			float tmp = size_sort/2.0;
			int blocksize = ceil(tmp);
			float tmp_f = 2 * blocksize;
			int N_pow2 = pow(2, ceil(log2(tmp_f)));
			tmp_f = N_pow2;
			steps = log2(tmp_f) * (log2(tmp_f) +1)/2;  
			int real_size = 2 * blocksize;


			int blocksize_dist = size_part - k;
			while(blocksize_dist > SM_SIZE_1) blocksize_dist = ceil(blocksize_dist / 2.0);


			dim3 BlockDistRec( blocksize_dist, 1, 1);
			dim3 BlockMergeHoriz( blocksize, 1, 1);

			int size_v2 = ppl - (blockInd + 1) * partsize;
			dim3 GridMergeVer(size_v2, batch_leaves_1, batch_leaves_2);

			ComputeRecDists <<< GridDistRec, BlockDistRec >>> (d_R, d_C, d_V, d_GId, d_Norms, k, ppl, blockInd, d_temp_knn, bl, sizebleaves, partsize);
			checkCudaErrors(cudaDeviceSynchronize());

			checkCudaErrors(cudaEventRecord(t6, 0));
			checkCudaErrors(cudaEventSynchronize(t6));
			
			PrecompSortIdsSparse(d_arr, d_arr_part, real_size, N_pow2, steps, copy_size);
			
			MergeHoriz <<< GridMergeHoriz, BlockMergeHoriz >>> (d_knn, d_knn_Id, k, ppl, blockInd, d_temp_knn, d_arr, d_arr_part, steps, d_GId, false, bl, sizebleaves, partsize); 
			checkCudaErrors(cudaDeviceSynchronize());
			checkCudaErrors(cudaEventRecord(t7, 0));
			checkCudaErrors(cudaEventSynchronize(t7));
			
      MergeVer_v2 <<< GridMergeVer, BlockMergeVer >>> (d_knn, d_knn_Id, k, ppl, blockInd, d_temp_knn, d_GId, false, bl, sizebleaves, partsize);
			checkCudaErrors(cudaDeviceSynchronize());
			checkCudaErrors(cudaEventRecord(t8, 0));
			checkCudaErrors(cudaEventSynchronize(t8));
			checkCudaErrors(cudaEventElapsedTime(&dt_tmp, t5, t6));
			dt5 += dt_tmp; 
			checkCudaErrors(cudaEventElapsedTime(&dt_tmp, t6, t7)); 
			dt6 += dt_tmp; 
			checkCudaErrors(cudaEventElapsedTime(&dt_tmp, t7, t8)); 
			dt7 += dt_tmp; 

		}
  
  }

  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaEventRecord(t9, 0));
  checkCudaErrors(cudaEventSynchronize(t9));
  checkCudaErrors(cudaEventElapsedTime(&dt1, t0, t1));
  checkCudaErrors(cudaEventElapsedTime(&dt2, t1, t2));
  checkCudaErrors(cudaEventElapsedTime(&dt3, t2, t3));
  checkCudaErrors(cudaEventElapsedTime(&dt4, t3, t4));
  checkCudaErrors(cudaEventElapsedTime(&dt8, t4, t9));
  checkCudaErrors(cudaEventElapsedTime(&dt9, t0, t9));

  //checkCudaErrors(cudaMemcpy(knn, d_knn, sizeof(float) * M * k, cudaMemcpyDeviceToHost));
  //checkCudaErrors(cudaMemcpy(knn_Id, d_knn_Id, sizeof(int) * M * k, cudaMemcpyDeviceToHost));
  

  checkCudaErrors(cudaFree(d_Norms));
  checkCudaErrors(cudaFree(d_temp_knn));
  checkCudaErrors(cudaFree(d_arr_part));
  checkCudaErrors(cudaFree(d_arr));

  checkCudaErrors(cudaEventDestroy(t0));
  checkCudaErrors(cudaEventDestroy(t1));
  checkCudaErrors(cudaEventDestroy(t2));
  checkCudaErrors(cudaEventDestroy(t3));
  checkCudaErrors(cudaEventDestroy(t4));
  checkCudaErrors(cudaEventDestroy(t5));
  checkCudaErrors(cudaEventDestroy(t6));
  checkCudaErrors(cudaEventDestroy(t7));
  checkCudaErrors(cudaEventDestroy(t8));
  checkCudaErrors(cudaEventDestroy(t9));
  
  printf("--------------- Timings ----------------\n");
  printf("\t Memory allocation :%.4f (%.4f %%) \n", dt1/1000, dt1/dt9);
  printf("\t Precomp sortId (vertical)  = %.4f (%.4f %%) \n", dt2/1000, dt2/dt9);
  printf("\t Computing norms = %.4f (%.4f %%) \n", dt3/1000, dt3/dt9);
  printf("\t Diagonal part = %.4f (%.4f %%) \n", dt4/1000, dt4/dt9);
  printf("\t Iterative part = %.4f (%.4f %%) \n", dt8/1000, dt8/dt9);
  printf("\t\t	 Compute Dists = %.4f (%.4f %%) \n", dt5/1000, dt5/dt9);
  printf("\t\t	 Merge Horizontally = %.4f (%.4f %%) \n", dt6/1000, dt6/dt9);
  printf("\t\t	 Merge Vertically  = %.4f (%.4f %%) \n", dt7/1000, dt7/dt9);
  printf("\t Total = %.4f \n", dt9/1000);
  printf("--------------- Memory usage ----------------\n");
  printf("\t Storing norms = %.4f GB \n", (m1-m2)/1e9);
  printf("\t Precomputing the sort indices = %.4f GB \n", (free-m1)/1e9);
	printf("\t Temporary storage = %.4f GB \n", (m2-m3)/1e9); 
  printf("----------------------------- End of leaf-knn -----------------------------\n\n");
  
}











