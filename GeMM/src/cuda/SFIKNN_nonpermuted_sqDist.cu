
#define SM_SIZE_1 1024
#define SM_SIZE_2 2048
#define SM_SIZE_SORT 8192

#include "FIKNN_sparse.h"

__global__ void ComputeNorms(int* R, int* C, float* V, int* G_Id, float* Norms, int ppl) {

  //int row = threadIdx.x + blockIdx.x * blockDim.x;
  int ind = threadIdx.x;
  int leafId_g = blockIdx.z * blockDim.y + blockIdx.y;
  for (int row = ind; row < ppl; row += blockDim.x){
    int g_rowId = leafId_g * ppl + row;
    //changed
  
    int g_Id = G_Id[g_rowId];

   
    int ind0_i = R[g_Id];
 
    int nnz = R[g_Id + 1] - ind0_i;
    float norm_i = 0.0;
   
    for (int n_i = 0; n_i < nnz; n_i += 1) {
      norm_i += V[ind0_i + n_i] * V[ind0_i + n_i];
    }
    Norms[g_Id] = norm_i;
  }
}

__global__ void ComputeTriDists(int* R, int* C, float* V, int* G_Id, float* Norms , int k_nn, float* KNN_dist_tmp, int ppl, int bl, int sizebleaves, int partsize) {

  int ind = threadIdx.x;
  int leafId_local = blockIdx.z * blockDim.y + blockIdx.y;
  int leafId_g = bl * sizebleaves + leafId_local;
  int block = blockIdx.x;


  //int size_block = k_nn * (k_nn + 1) /2;
  int size_block = partsize * (partsize + 1) /2;
  
  for (int elem = ind; elem < size_block; elem += blockDim.x){

    //float tmp = -8 * elem + 4 * k_nn * (k_nn+1) - 7;
    float tmp = -8 * elem + 4 * partsize * (partsize + 1) - 7;
    int rowId = sqrt(tmp)/2.0 - 0.5;
    //rowId = k_nn - 1 - rowId;
    rowId = partsize - 1 - rowId;
    //int colId = elem + rowId - k_nn * (k_nn + 1) / 2 + (k_nn - rowId) * ((k_nn - rowId) + 1)/2;
    int colId = elem + rowId - partsize * (partsize + 1) / 2 + (partsize - rowId) * ((partsize - rowId) + 1)/2;

    float c_tmp = 0.0;

    //int g_rowId = leafId_g * ppl + block * k_nn + rowId;
    //int g_colId = leafId_g * ppl + block * k_nn + colId;
    int g_rowId = leafId_g * ppl + block * partsize + rowId;
    int g_colId = leafId_g * ppl + block * partsize + colId;
   
    //changed 
    int perm_i = G_Id[g_rowId];
    int perm_j = G_Id[g_colId];

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
    c_tmp = (c_tmp > 1e-8) ? sqrt(c_tmp) : 0.0;
    
    // changed 
    
    int gid_pt = leafId_local * ppl + block * partsize + rowId;
    int gid_pt_T = leafId_local * ppl + block * partsize + colId;
    int ind_knn = gid_pt * partsize + colId;
    int ind_knn_T = gid_pt_T * partsize + rowId;
    
    KNN_dist_tmp[ind_knn] = c_tmp;
    if (colId > rowId) KNN_dist_tmp[ind_knn_T] = c_tmp;
     
  }
  
}



__global__ void ComputeTriDists_last(int* R, int* C, float* V, int* G_Id, float* Norms , int k_nn, float* KNN_dist_tmp, int ppl, int rem_len , int blockId, int bl, int sizebleaves, int partsize) {




  int ind = threadIdx.x;
  //int leaf_id_g = blockIdx.z * blockDim.y + blockIdx.y;
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
    int perm_i = G_Id[g_rowId];
    int perm_j = G_Id[g_colId];

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


    int gid_pt = leafId_local * ppl + block * partsize + rowId;
    int gid_pt_T = leafId_local * ppl + block * partsize + colId;
    int ind_knn = gid_pt * partsize + colId;
    int ind_knn_T = gid_pt_T * partsize + rowId;

    //if (gid_pt == 4679) printf("last D[%d] = %.4f , write at %d \n", colId, c_tmp, ind_knn); 
    //if (gid_pt_T == 4679) printf("last D[%d] = %.4f , write at %d \n", rowId, c_tmp, ind_knn_T);
 
    KNN_dist_tmp[ind_knn] = c_tmp;
    if (colId > rowId) KNN_dist_tmp[ind_knn_T] = c_tmp;

    for (int row_tmp = 0; row_tmp<rem_len; row_tmp++){
      for (int q = ind + rem_len; q < partsize; q += blockDim.x){
        gid_pt = leafId_local * ppl + block * partsize + row_tmp;
        ind_knn = gid_pt * partsize + q;
        KNN_dist_tmp[ind_knn] = 1e30;
      } 
    } 


  }

}



__global__ void ComputeRecDists(int* R, int* C, float* V, int* G_Id, float* Norms, int k_nn, int ppl, int blockInd, float* KNN_Dist_tmp, int bl, int sizebleaves, int partsize, int maxnnz_pow2, int numsqblocks_i, int numsqblocks_j, int sizesqblocks) {

  
  __shared__ int SM_I[6144];
  __shared__ int SM_J[6144];
  
  int ind = blockIdx.x;

  int block_I = ind / numsqblocks_j;
  int block_J = ind - block_I * numsqblocks_j;
  
  int j = threadIdx.x;
  int row_l = j / sizesqblocks;
  int col_l = j - row_l * sizesqblocks;
 
  int leafId_local = blockIdx.z * blockDim.y + blockIdx.y;
  int leafId_g =  bl * sizebleaves + leafId_local;
  
  int size_part = ppl - (partsize) * (blockInd+1); 
  
  int rowId_leaf = partsize * blockInd + sizesqblocks * block_I + row_l;
  int g_rowId_I = leafId_g * ppl + rowId_leaf;
  int colId_leaf = partsize * (blockInd+1) + block_J * sizesqblocks + col_l;
  int g_rowId_J = leafId_g * ppl + colId_leaf;
 
		
  //changed 

  int perm_i,  ind0_i, ind1_i, nnz_i, shift_i;	
  int perm_j,  ind0_j, ind1_j, nnz_j, shift_j;
	float norm_ij=0.0;

  if (rowId_leaf < ppl){	

    perm_i = G_Id[g_rowId_I];
		
    ind0_i = R[perm_i];
		ind1_i = R[perm_i+1];
		
    norm_ij += Norms[perm_i];
		nnz_i = ind1_i - ind0_i;
		shift_i = maxnnz_pow2 * row_l;
	
  }
  
  if (colId_leaf < ppl){	 
		perm_j = G_Id[g_rowId_J];
		ind0_j = R[perm_j];
		ind1_j = R[perm_j+1];
		nnz_j = ind1_j - ind0_j;
		norm_ij += Norms[perm_j];
		shift_j = maxnnz_pow2 * col_l;
  }
  
  if (rowId_leaf < ppl){ 
		for (int n_i = col_l; n_i< nnz_i; n_i += sizesqblocks) SM_I[shift_i + n_i] = C[ind0_i + n_i];
  }
  if (colId_leaf < ppl){
		for (int n_j = row_l; n_j< nnz_j; n_j += sizesqblocks) SM_J[shift_j + n_j] = C[ind0_j + n_j];
	}
	__syncthreads();

	float c_tmp = 0.0;
	int size_tmp = size_part;
	int ind_tmp = leafId_local * partsize * size_tmp + (block_I * sizesqblocks + row_l) * size_tmp + colId_leaf - (partsize) * (blockInd+1);
  if (rowId_leaf < ppl && colId_leaf < ppl){					
		int tmp_0, tmp_1, ind_jk, k, ret, testInd;
				
		ret = 0;
		testInd = 0;
				
		// loop over the elements of j
		
		if (nnz_i >0 && nnz_j > 0){
			for (int pos_k = 0; pos_k < nnz_j; pos_k++){
					
				//k = C[ind0_j + pos_k];
				//k = C_Y[pos_k];
				k = SM_J[shift_j + pos_k];	
				// Binary search
			
				for (int l = nnz_i - ret; l > 1; l -= floorf(l/2.0)){

					tmp_0 = ret + l;
					tmp_1 = nnz_i - 1;
							
					testInd = (tmp_0 < tmp_1) ? tmp_0 : tmp_1;
							
					ret = (SM_I[shift_i + testInd] <= k) ? testInd : ret;
					//ret = (C[ind0_i + testInd] <= k) ? testInd : ret;
				}

				tmp_0 = ret + 1;
				tmp_1 = nnz_i - 1;
						
				testInd = (tmp_0 < tmp_1 ) ? tmp_0 : tmp_1;
				ret = (SM_I[shift_i + testInd] <= k) ? testInd : ret;
				//ret = (C[ind0_i +testInd] <= k) ? testInd : ret;
						
				ind_jk = (SM_I[shift_i + ret] == k) ? ret : -1;
				//ind_jk = (C[ind0_i +ret] == k) ? ret : -1;
				c_tmp += (ind_jk != -1) ? V[ind0_j + pos_k] * V[ind0_i + ind_jk] : 0;
				} 
		}
		//int ind_tmp = leafId_local * partsize * size_tmp + (block_I * sizesqblocks + row_l) * size_tmp + colId_leaf - (partsize) * (blockInd+1);
		//int ind_tmp = leafId_local * partsize * size_tmp + (block_I * sizesqblocks + row_l) * size_tmp + (block_J * sizesqblocks + col_l);
				 
		c_tmp = -2 * c_tmp + norm_ij;
		c_tmp = (c_tmp > 1e-8) ? sqrt(c_tmp) : 0.0;
				
    KNN_Dist_tmp[ind_tmp] = c_tmp;
  }
  //if (leafId_g == 0 && blockInd == 0) printf("Compute for row = %d , col = %d , (%d, %d) \n", G_Id[g_rowId_I], G_Id[g_rowId_J], g_rowId_I, g_rowId_J);
  //if (G_Id[g_rowId_I] == 0) printf("dist tmp write %.4f at %d \n", c_tmp, ind_tmp);
  //if (G_Id[g_rowId_J] == 0) printf("dist tmp write %.4f at %d \n", c_tmp, ind_tmp);
 

}




__global__ void MergeHoriz(float* KNN, int* KNN_Id, int k_nn, int ppl, int blockInd, float* d_temp_knn, int* sort_arr, int* sort_arr_part, int steps, int* G_Id, bool init, int bl, int sizebleaves, int partitionsize){

   
  __shared__ float SM_dist[SM_SIZE_2];
  __shared__ int SM_Id[SM_SIZE_2];



  int j = threadIdx.x;
  int row_l = blockIdx.x;
  int leafId_local = blockIdx.z * blockDim.y + blockIdx.y;
  int leafId_g = bl * sizebleaves + leafId_local;
  

  //int size_part = ppl - (k_nn) * blockInd;
  //int size_part = ppl - (k_nn) * blockInd;
  int size_part = ppl - (partitionsize) * blockInd;
  int size_sort = 2 * blockDim.x;

  //int rowId_leaf = k_nn * blockInd + row_l;
  int rowId_leaf = partitionsize * blockInd + row_l;

  
  for (int n=j; n < SM_SIZE_2; n += blockDim.x){
    SM_dist[n] = 1e30; 
    SM_Id[n] = -1;
  }

  float tmp_f;
  int tmp_i;
  int ind_sort;
   
  int num_batches = (size_part) / (size_sort - k_nn);
  //int num_batches = size_part / (size_sort - partitionsize);
  int ind_pt = G_Id[leafId_g * ppl + rowId_leaf];
  
  for (int col_batch = 0; col_batch < num_batches; col_batch++){
    
    for (int j_tmp = j; j_tmp < size_sort; j_tmp += blockDim.x){
      
      int colId_leaf = (col_batch == 0) ? partitionsize * (blockInd+1) + j_tmp - k_nn: partitionsize * (blockInd+1) + (col_batch)* (size_sort - k_nn) + j_tmp - k_nn;
      
      if (col_batch == 0 && j_tmp < k_nn){
        
        int ind_read = ind_pt * k_nn + j_tmp;
        SM_dist[j_tmp] = KNN[ind_read];
        SM_Id[j_tmp] = KNN_Id[ind_read];
        //if (ind_pt == 0) printf("read merge horiz, D[%d] = %.4f , %d , read %d \n", j_tmp, SM_dist[j_tmp], SM_Id[j_tmp], ind_read);
     
      } else if (colId_leaf < ppl && j_tmp >= k_nn){

        int size_tmp = size_part - partitionsize;
        //int ind_tmp = leafId_local * k_nn * size_tmp + row_l * size_tmp + colId_leaf - (k_nn) * (blockInd+1);
        int ind_tmp = leafId_local * partitionsize * size_tmp + row_l * size_tmp + colId_leaf - (partitionsize) * (blockInd+1);
        int g_colId_J = leafId_g * ppl + colId_leaf;
        //if (ind_tmp < 0) printf("leafId_local = %d , partitionsize = %d , size_tmp = %d , row_l = %d , colId_leaf = %d , blockInd = %d \n", leafId_local, partitionsize, size_tmp, row_l, colId_leaf, blockInd); 
        SM_dist[j_tmp] = d_temp_knn[ind_tmp];
        SM_Id[j_tmp] = G_Id[g_colId_J];
      } 
      
        
      //if (G_Id[leafId_g * ppl + rowId_leaf] == 0 && j_tmp < 256) printf("col_batch = %d , D[%d] = %.4f , at %d  \n", col_batch, j_tmp, SM_dist[j_tmp], SM_Id[j_tmp]);
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


  __syncthreads();
  //if (G_Id[leafId_g * ppl + rowId_leaf] == 0 && j < 256) printf("sorted col_batch = %d , D[%d] = %.4f , at %d  \n", col_batch, j, SM_dist[j], SM_Id[j]);
  }
  for (int j_tmp = j; j_tmp < k_nn; j_tmp += blockDim.x){ 
    if (j_tmp < k_nn){
      int ind_pt = leafId_g * ppl + rowId_leaf;
      int write_ind = G_Id[ind_pt] * k_nn + j_tmp;
      KNN[write_ind] = SM_dist[j_tmp];
      KNN_Id[write_ind] = SM_Id[j_tmp];
      //if (G_Id[ind_pt] == 0) printf("sorted D[%d] = %.4f , at %d , write at %d  \n", j_tmp, SM_dist[j_tmp], SM_Id[j_tmp], write_ind);
    }
  } 
}



__global__ void MergeVer(float* KNN, int* KNN_Id, int k_nn, int ppl, int blockInd, float* d_temp_knn, int* sort_arr, int* sort_arr_part, int steps, int* G_Id, bool init, int bl, int sizebleaves, int partitionsize){

  __shared__ float SM_dist[SM_SIZE_1];
  __shared__ int SM_Id[SM_SIZE_1];


  int j = threadIdx.x;

  int col = blockIdx.x;
  int leafId_local = blockIdx.z* blockDim.y + blockIdx.y;
  int leafId_g = bl * sizebleaves + leafId_local;
  int colId_leaf = (init) ? col : col + partitionsize * (blockInd + 1);
  int size_part = (init) ? ppl : ppl - (blockInd + 1) * partitionsize;
	
  int ind_pt_knn = leafId_g * ppl + colId_leaf;
	int ind_pt_knn_g = G_Id[ind_pt_knn];
  for (int ind = j; ind < 2 * partitionsize; ind += blockDim.x){
    SM_dist[ind] = 1e30;
    SM_Id[ind] = -1;
  }
  __syncthreads();

  for (int j_tmp = j; j_tmp < partitionsize; j_tmp += blockDim.x){
		
    int ind_tmp = (init) ? leafId_local * ppl * partitionsize + col * partitionsize + j_tmp : leafId_local * partitionsize * size_part + j_tmp * size_part + col;
		SM_dist[j_tmp] = d_temp_knn[ind_tmp];
		//int block = col / k_nn;
		int block = col / partitionsize;
		int rowId_g = (init) ? leafId_g * ppl + block * partitionsize + j_tmp : leafId_g * ppl + partitionsize * blockInd + j_tmp;
		SM_Id[j_tmp] = G_Id[rowId_g];

 
		int ind_knn = ind_pt_knn_g * k_nn + j_tmp;
		//SM_dist[j + k_nn] = KNN[ind_knn];
		//SM_Id[j + k_nn] = KNN_Id[ind_knn];
    //if (ind_pt_knn_g == 0) printf("read tmp init = %d , blockInd = %d , Ver D[%d] = %.4f , %d , read from %d \n", init , blockInd, j_tmp, SM_dist[j_tmp], SM_Id[j_tmp], ind_tmp);
		if (j_tmp < k_nn){
       
			SM_dist[j_tmp + partitionsize] = KNN[ind_knn];
			SM_Id[j_tmp + partitionsize] = KNN_Id[ind_knn];
      //if (ind_pt_knn_g == 0) printf("read neighbor ver init = %d , blockInd = %d , Ver D[%d] = %.4f , %d , read from %d \n", init , blockInd, j_tmp + partitionsize, KNN[ind_knn], KNN_Id[ind_knn], ind_knn);
		} else {
      SM_dist[j_tmp + partitionsize] = 1e30;
      SM_Id[j_tmp + partitionsize] = -1;
    }
    //if (ind_pt_knn_g == 0) printf("init = %d , blockInd = %d , Ver D[%d] = %.4f , %d \n", init , blockInd, j_tmp + partitionsize, SM_dist[j_tmp + partitionsize], SM_Id[j_tmp+partitionsize]);
  }
	__syncthreads();

  for (int j_tmp = j; j_tmp < 2 * partitionsize; j_tmp += blockDim.x){
    //if (ind_pt_knn_g == 0 && blockInd < 2) printf(" blockInd = %d , D[%d] = %.4f , at %d \n",blockInd, j_tmp, SM_dist[j_tmp], SM_Id[j_tmp]);
  }
  
  for (int j_tmp = j; j_tmp < partitionsize; j_tmp += blockDim.x){ 
    int index = SM_Id[j_tmp];
    for (int ind_check = 0; ind_check < k_nn; ind_check++){
      if (index == SM_Id[ind_check + partitionsize]){
        SM_dist[j_tmp] = 1e30;
        SM_Id[j_tmp] = -1;
        break;
      }
    }
  }
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

   
  if (j < k_nn){ 
		int ind_pt_knn = leafId_g * ppl + colId_leaf;
		int ind_pt_knn_g = G_Id[ind_pt_knn];
		int ind_knn = ind_pt_knn_g * k_nn + j;
  	KNN[ind_knn] = SM_dist[j];
		KNN_Id[ind_knn] = SM_Id[j];
	  //if (ind_pt_knn_g == 0) printf("Ver write init = %d, D[%d] = %.4f , at %d , write at %d \n",init, j, KNN[ind_knn], KNN_Id[ind_knn], ind_knn);
  } 
  

}
















/*
__global__ void knn_kernel_B(float* KNN, int* KNN_Id, int k_nn, int ppl, int blockInd, float* d_temp_knn, int* G_Id, bool init, int bl, int sizebleaves){

  __shared__ float SM_dist[SM_SIZE_1];
  __shared__ int SM_Id[SM_SIZE_1];

  int tid = threadIdx.x;

  int col = blockIdx.x;
  int leafId_local = blockIdx.z * blockDim.y + blockIdx.y;
  int leafId_g = bl * sizebleaves + leafId_local;
  
  int colId_leaf = (init) ? col : col + k_nn * (blockInd + 1);
  int size_part = (init) ? ppl : ppl - (blockInd + 1) * (k_nn);
  if (tid < k_nn){
    //changed 
    int ind_tmp = (init) ? leafId_local * ppl * k_nn + col * k_nn + tid : leafId_local * k_nn * size_part + tid * size_part + col;
    SM_dist[tid] = (colId_leaf < ppl) ? d_temp_knn[ind_tmp] : 1e30;
   
    int block = col / k_nn;
    int rowId_g = (init) ? leafId_g * ppl + block * k_nn + tid : leafId_g * ppl + k_nn * blockInd + tid;  
    SM_Id[tid] = (colId_leaf < ppl) ? G_Id[rowId_g] : -1;
  } else {

    int ind_pt_knn = leafId_g * ppl + colId_leaf;
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
  
  int ind_pt = leafId_g * ppl + colId_leaf;    
  int ind_pt_g = G_Id[ind_pt];
  int write_ind = ind_pt_g * k_nn + tid;
   
  if (tid < k_nn) {
    KNN[write_ind] = SM_dist[tid];
    KNN_Id[write_ind] = SM_Id[tid];
  }

}
*/

void PrecompSortIds(int* d_arr, int* d_arr_part, int N_true, int N_pow2, int steps, int copy_size){

  
  
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
  //checkCudaErrors(cudaDeviceSynchronize());
  
  free(arr);
  free(arr_part); 
  free(tracker); 
  
}



void sfi_leafknn(int *R, int *C, float *V, int *G_Id, int M, int leaves, int k, float *knn, int *knn_Id, int maxnnz){



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



  int ppl = M/leaves;

  int partitionsize = (k > 32) ? k : 32;
  int num_blocks_tri = ppl / partitionsize;
  int rem_len = (num_blocks_tri * partitionsize < ppl) ? ppl - num_blocks_tri * partitionsize : 0;
 
  int C_len = R[M];
  int t_b = (ppl > SM_SIZE_1) ? SM_SIZE_1 : ppl;
  float maxnnz_f = maxnnz;
  int logmaxnnz = ceil(log2(maxnnz_f));
  int maxnnz_pow2 = 1 << logmaxnnz;
  
  int num_splits = 1;
  while (leaves > num_splits * 65535) num_splits *= 2;

  int batch_leaves_1 = (leaves > 64000) ? leaves / num_splits : leaves;
  int batch_leaves_2 = (leaves > 64000) ? num_splits : 1;

  int sizesqblocks = 12288/(2*maxnnz_pow2);
  sizesqblocks = (sizesqblocks > 32) ? 32 : sizesqblocks;
  int numsqblocks_i = ceil(partitionsize/(sizesqblocks*1.0));

  dim3 BlockNorm(t_b, 1, 1);
  dim3 GridNorm(1, batch_leaves_1, batch_leaves_2);

  
  int verbose = 1;


  if (verbose) printf("----------------------------- Start of sfiknn ----------------------------- \n\n");

  float *d_Norms;

  int size_tri = partitionsize;
  int blockDim_tri = size_tri * (size_tri + 1)/2;
  if (blockDim_tri > SM_SIZE_1) blockDim_tri = SM_SIZE_1;

  int size_tri_last = (rem_len > 32) ? 32 : rem_len;
  int blockDim_tri_last = size_tri_last * (size_tri_last + 1)/2;
  if (blockDim_tri_last > SM_SIZE_1) blockDim_tri_last = SM_SIZE_1;

  dim3 BlockDistTri(blockDim_tri, 1, 1);
  dim3 GridDistTri(num_blocks_tri, batch_leaves_1, batch_leaves_2);
  dim3 BlockDistTri_last(blockDim_tri_last, 1, 1);
  dim3 GridDistTri_last(1, batch_leaves_1, batch_leaves_2);

	dim3 BlockDistRec(sizesqblocks* sizesqblocks, 1, 1);

  dim3 GridMergeHoriz(partitionsize, batch_leaves_1, batch_leaves_2);

  int size_v_block_reduced = (k + partitionsize)/2;
  dim3 BlockMergeVer(size_v_block_reduced, 1, 1);
 
  dim3 BlockSortGIds(t_b, 1, 1);
  dim3 GridSortGIds(leaves, 1, 1);
 
  printf("=======================\n");
  printf(" Num points = %d \n", M);
  printf(" pt/leaf = %d \n", ppl);
  printf(" Leaves = %d \n", leaves);
  printf(" K = %d \n", k);
  printf(" PartSize = %d \n", partitionsize);

  printf(" dim BlockThreads  Norms = (%d , %d, %d) \n", BlockNorm.x, BlockNorm.y, BlockNorm.z);
  printf(" dim GridThreads Norms = (%d , %d, %d) \n", GridNorm.x, GridNorm.y, GridNorm.z);
  printf(" dim BlockThreads Diagonal Distances = (%d , %d, %d) \n", BlockDistTri.x, BlockDistTri.y, BlockDistTri.z);
  printf(" dim GridThreads Diagonal Distances = (%d , %d, %d) \n", GridDistTri.x, GridDistTri.y, GridDistTri.z);
  printf(" dim BlockThreads Diagonal Distances last = (%d , %d, %d) \n", BlockDistTri_last.x, BlockDistTri_last.y, BlockDistTri_last.z);
  printf(" dim GridThreads Diagonal Distances last = (%d , %d, %d) \n", GridDistTri_last.x, GridDistTri_last.y, GridDistTri_last.z);
  printf(" dim BlockThreads Distance Horiz = (%d , %d, %d) \n", BlockDistRec.x, BlockDistRec.y, BlockDistRec.z);
  printf(" dim GridThreads MergeVer = (%d , %d, %d) \n", GridMergeHoriz.x, GridMergeHoriz.y, GridMergeHoriz.z);
  printf(" dim BlockMerge MergeVer = (%d , %d, %d) \n", BlockMergeVer.x, BlockMergeVer.y, BlockMergeVer.z);
  



  int *d_arr, *d_arr_part, *d_arr_v, *d_arr_part_v;
  float SM_SIZE_2_f = SM_SIZE_2;
  int numsteps = log2(SM_SIZE_2_f) *(log2(SM_SIZE_2_f)+1) /2;

  int copy_size = (ppl) * numsteps;
  float tmp = 2 * partitionsize;
  int numsteps_ver = log2(tmp) * (log2(tmp)+1)/2;
  int copy_size_ver = (2 * partitionsize) * numsteps_ver;

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
  checkCudaErrors(cudaMemcpy(d_knn, knn, sizeof(float) * M * k, cudaMemcpyHostToDevice)); 
  checkCudaErrors(cudaMemcpy(d_knn_Id, knn_Id, sizeof(int) * M * k, cudaMemcpyHostToDevice)); 

  cudaMemGetInfo(&free, &total);
  checkCudaErrors(cudaMalloc((void **) &d_arr, sizeof(int) * copy_size));
  checkCudaErrors(cudaMalloc((void **) &d_arr_part, sizeof(int) * copy_size));

  checkCudaErrors(cudaMalloc((void **) &d_arr_v, sizeof(int) * copy_size_ver));
  checkCudaErrors(cudaMalloc((void **) &d_arr_part_v, sizeof(int) * copy_size_ver));


  checkCudaErrors(cudaMemset(d_arr, 0, sizeof(int) * copy_size));
  checkCudaErrors(cudaMemset(d_arr_part, 0, sizeof(int) * copy_size));
  checkCudaErrors(cudaMemset(d_arr_v, 0, sizeof(int) * copy_size_ver));
  checkCudaErrors(cudaMemset(d_arr_part_v, 0, sizeof(int) * copy_size_ver));
  cudaMemGetInfo(&m1, &total);



  checkCudaErrors(cudaEventRecord(t1, 0));
   
  int size_sort_ver = k + partitionsize;
  int size_sort_ver_pow2 = 2*partitionsize;
  PrecompSortIds(d_arr_v, d_arr_part_v, size_sort_ver, size_sort_ver_pow2, numsteps_ver, copy_size_ver);


  //sort_GIds <<< GridSortGIds, BlockSortGIds >>> (d_GId, ppl);
  
  checkCudaErrors(cudaEventRecord(t2, 0));

  float * d_temp_knn;
  checkCudaErrors(cudaMalloc((void **) &d_Norms, sizeof(float) * M));

  cudaMemGetInfo(&m2, &total);
  float size_tmp = sizeof(float) * M * partitionsize;
  int bleaves = (size_tmp > m2) ? log2(size_tmp / m2) : 0;
  int numbleaves = 1 << bleaves;
  int sizebleaves = leaves / numbleaves; 
  printf(" Num BatchLeaves = %d \n", numbleaves);
  printf(" Size BatchLeaves = %d \n", sizebleaves);
  printf("=======================\n");

  

  
  checkCudaErrors(cudaMalloc((void **) &d_temp_knn, sizeof(float) * sizebleaves * ppl * partitionsize));
  cudaMemGetInfo(&m3, &total);

  int steps;

  ComputeNorms <<< GridNorm, BlockNorm >>>(d_R, d_C, d_V, d_GId, d_Norms, ppl);
  checkCudaErrors(cudaDeviceSynchronize());

  checkCudaErrors(cudaEventRecord(t3, 0));
  dt5 = 0.0; 
  dt6 = 0.0; 
  dt7 = 0.0; 
  for (int bl = 0; bl < numbleaves; bl++){

		ComputeTriDists <<< GridDistTri, BlockDistTri >>>(d_R, d_C, d_V, d_GId, d_Norms, k, d_temp_knn, ppl, bl, sizebleaves, partitionsize);
		checkCudaErrors(cudaDeviceSynchronize());
  
		if (rem_len > 0) {
			ComputeTriDists_last <<< GridDistTri_last, BlockDistTri_last >>>(d_R, d_C, d_V, d_GId, d_Norms, k, d_temp_knn, ppl, rem_len, num_blocks_tri, bl, sizebleaves, partitionsize);
			checkCudaErrors(cudaDeviceSynchronize());
		}


		int size_v = ppl;
		dim3 GridMergeVer(size_v, batch_leaves_1, batch_leaves_2);
    
		MergeVer <<< GridMergeVer, BlockMergeVer >>> (d_knn, d_knn_Id, k, ppl, 0, d_temp_knn, d_arr_v, d_arr_part_v, numsteps_ver, d_GId, true, bl, sizebleaves, partitionsize);
		checkCudaErrors(cudaDeviceSynchronize());
		checkCudaErrors(cudaEventRecord(t4, 0));
  
		int num_iters = (rem_len > 0) ? num_blocks_tri : num_blocks_tri - 1;
		for (int blockInd = 0; blockInd < num_iters; blockInd++){

		  checkCudaErrors(cudaEventRecord(t5, 0));	
			
      int size_part = ppl - (blockInd +1) * partitionsize;
			int size_sort = size_part + k;

			while (size_sort > SM_SIZE_2) size_sort = ceil((size_sort + k)/2);
			float tmp = size_sort/2.0;
			int blocksize = ceil(tmp);
			float tmp_f = 2 * blocksize;
			int N_pow2 = pow(2, ceil(log2(tmp_f)));
			tmp_f = N_pow2;
			steps = log2(tmp_f) * (log2(tmp_f) +1)/2;
			


			int real_size = 2 * blocksize;
			PrecompSortIds(d_arr, d_arr_part, real_size, N_pow2, steps, copy_size);

			//int blocksize_dist = size_part - partitionsize;
			int blocksize_dist = size_part;
			//while(blocksize_dist > SM_SIZE_1) blocksize_dist = ceil(blocksize_dist / 2.0);

      int numsqblocks_j = ceil(blocksize_dist/(sizesqblocks * 1.0));

			dim3 GridDistRec( numsqblocks_i * numsqblocks_j, batch_leaves_1, batch_leaves_2);

			dim3 BlockMergeHoriz( blocksize, 1, 1);

      
      //if (blockInd == 0) printf("size_part = %d , size_sort = %d , num_batch = %d \n", size_part, size_sort, (size_part+k)/size_sort); 
			int size_v2 = ppl - (blockInd + 1) * partitionsize;
			dim3 GridMergeVer(size_v2, batch_leaves_1, batch_leaves_2);
      /*      
      printf("GridDist = %d, %d , %d \n", GridDistRec.x, GridDistRec.y, GridDistRec.z);
      printf("BlockMergeHoriz = %d, %d , %d \n",BlockMergeHoriz.x, BlockMergeHoriz.y, BlockMergeHoriz.z);
      printf("GridMergeVer  = %d, %d , %d \n", BlockMergeHoriz.x, BlockMergeHoriz.y, BlockMergeHoriz.z);
      */
			ComputeRecDists <<< GridDistRec, BlockDistRec >>> (d_R, d_C, d_V, d_GId, d_Norms, k, ppl, blockInd, d_temp_knn, bl, sizebleaves, partitionsize, maxnnz_pow2, numsqblocks_i, numsqblocks_j, sizesqblocks);
    	checkCudaErrors(cudaDeviceSynchronize());
		  checkCudaErrors(cudaEventRecord(t6, 0));
			
			MergeHoriz <<< GridMergeHoriz, BlockMergeHoriz >>> (d_knn, d_knn_Id, k, ppl, blockInd, d_temp_knn, d_arr, d_arr_part, steps, d_GId, false, bl, sizebleaves, partitionsize); 
			checkCudaErrors(cudaDeviceSynchronize());
			checkCudaErrors(cudaEventRecord(t7, 0));
		
			MergeVer <<< GridMergeVer, BlockMergeVer >>> (d_knn, d_knn_Id, k, ppl, blockInd, d_temp_knn, d_arr_v, d_arr_part_v, numsteps_ver, d_GId, false,bl, sizebleaves, partitionsize);
			checkCudaErrors(cudaDeviceSynchronize());
			checkCudaErrors(cudaEventRecord(t8, 0));
      checkCudaErrors(cudaEventElapsedTime(&dt_tmp, t5, t6));
      dt5 += dt_tmp;
      checkCudaErrors(cudaEventElapsedTime(&dt_tmp, t6, t7));
      dt6 += dt_tmp;
      checkCudaErrors(cudaEventElapsedTime(&dt_tmp, t7, t8));
      dt7 += dt_tmp;
    }
  
  }



  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaEventSynchronize(t9));
  checkCudaErrors(cudaEventRecord(t9, 0));
  checkCudaErrors(cudaEventElapsedTime(&dt1, t0, t1));
  checkCudaErrors(cudaEventElapsedTime(&dt2, t1, t2));
  checkCudaErrors(cudaEventElapsedTime(&dt3, t2, t3));
  checkCudaErrors(cudaEventElapsedTime(&dt4, t3, t4));
  checkCudaErrors(cudaEventElapsedTime(&dt8, t4, t9));
  checkCudaErrors(cudaEventElapsedTime(&dt9, t0, t9));

  //checkCudaErrors(cudaMemcpy(knn, d_knn, sizeof(float) * M * k, cudaMemcpyDeviceToHost));
  //checkCudaErrors(cudaMemcpy(knn_Id, d_knn_Id, sizeof(int) * M * k, cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(knn, d_knn, sizeof(float) * M * k, cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(knn_Id, d_knn_Id, sizeof(int) * M * k, cudaMemcpyDeviceToHost));


  checkCudaErrors(cudaFree(d_Norms));
  checkCudaErrors(cudaFree(d_temp_knn));
  checkCudaErrors(cudaFree(d_arr_part));
  checkCudaErrors(cudaFree(d_arr));
  checkCudaErrors(cudaFree(d_arr_part_v));
  checkCudaErrors(cudaFree(d_arr_v));

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
  //cudaMemGetInfo(&free, &total);
  printf("--------------- Timings ----------------\n");
  printf("Memory allocation = %.4f (%.4f %%) \n", dt1/1e3, dt1/dt9);
  printf("Precomp sortId (vertical) = %.4f (%.4f %%) \n", dt2/1e3, dt2/dt9);
  printf("Computing norms = %.4f (%.4f %%) \n", dt3/1e3, dt3/dt9);
  printf("Diagonal part = %.4f (%.4f %%) \n", dt4/1e3, dt4/dt9);
  printf("Iterative part = %.4f (%.4f %%) \n", dt8/1e3, dt8/dt9);
  printf("\tCompute Dists = %.4f (%.4f %%) \n", dt5/1e3, dt5/dt9);
  printf("\tMerge Horizontally = %.4f (%.4f %%) \n", dt6/1e3, dt6/dt9);
  printf("\tMerge Vertically  = %.4f (%.4f %%) \n", dt7/1e3, dt7/dt9);
  printf("Total = %.4f \n", dt9/1e3);
  printf("--------------- Memory usage ----------------\n");
  printf("Storing norms = %.4f GB \n", (m1-m2)/1e9);
  printf("Precomputing the sort indices = %.4f GB \n", (free-m1)/1e9);
  printf("Temporary storage = %.4f GB \n", (m2-m3)/1e9);
  printf("----------------------------- End of leaf-knn -----------------------------\n\n");

}











