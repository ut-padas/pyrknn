
#include "knn_seq.h"



void f_knnSeq(int *R, int* C, float* V, int* G_Id, float* K, int* K_Id, int knn, int leaf_id, int pt_id, int leaf_size){

	int start = leaf_id * leaf_size;


 
  int nnz_i, nnz_j;
  float c;
  int ret;
  int testInd;
  float norm;
  std::pair<float, int> pairt[leaf_size];
  int *Id_tmp;
  float *k_tmp;
  Id_tmp = (int *)malloc(sizeof(int) * leaf_size);
  k_tmp = (float *)malloc(sizeof(float) * leaf_size);
  int ind0_i, ind0_j, tmp_0, tmp_1, k, ind_jk;
  
  //for (int i=0; i< leaf_size; i++){
    int i = pt_id;
    //printf("seq for %d \n", i);
    norm = 0.0;
    float norm_i = 0;
    ind0_i = R[G_Id[start + i]];
    nnz_i = R[G_Id[start + i] + 1] - ind0_i;
    //norm += V[ind0_i + i] * V[ind0_i + i];
    for (int s = 0; s < nnz_i; s++) norm_i += V[ind0_i + s] * V[ind0_i + s];
    for (int j=0; j < leaf_size; j++){
      ind0_j = R[G_Id[start + j]];
      nnz_j = R[G_Id[start + j] + 1] - ind0_j;
      
      
      norm = norm_i;
      for (int s = 0; s < nnz_j; s++) norm += V[ind0_j + s] * V[ind0_j + s];

      //norm = V[ind0_j + j] * V[ind0_j + j] + V[ind0_i + i] * V[ind0_i + i];
      c = 0;
      ret = 0;
      testInd = 0;
      ind_jk = -1; 
      for (int pos = 0; pos < nnz_j; pos ++){
        k = C[ind0_j + pos];

        for (int l=nnz_i-ret; l > 1; l/=2){
          tmp_0 = ret+l;
          tmp_1 = nnz_i - 1;
          testInd = (tmp_0 < tmp_1) ? tmp_0: tmp_1;
          ret = (C[ind0_i + testInd] <= k) ? testInd : ret ;
        }
        tmp_0 = ret+1;
        tmp_1 = nnz_i-1;
        testInd = (tmp_0 < tmp_1) ? tmp_0: tmp_1;
        ret = (C[ind0_i + testInd] <= k) ? testInd : ret;
        ind_jk = (C[ind0_i + ret] == k) ? ret : -1;
        c += (ind_jk != -1) ? V[ind0_j + pos]*V[ind0_i + ind_jk] : 0;
      }
      c = -2*c + norm; 
      c = (c>0) ? sqrt(c): 0.0;
    
      k_tmp[j] = c;
      Id_tmp[j] = G_Id[leaf_size * leaf_id + j];
      //printf("|| %d - %d || = %.4f , loc = %d ", )
    
    
    
    
  }
   
  for (int s = 0; s < leaf_size; s++){
    //printf("pt %d , %.4f \n", i , k_tmp[s]);  
    pairt[s].first = k_tmp[s];
    pairt[s].second = Id_tmp[s];
  
  }

  std::sort(pairt, pairt + leaf_size);

  for (int s = 0; s< knn; s++){

    K[s] = pairt[s].first;
    K_Id[s] = pairt[s].second;

  }


}
