
#include "knn_seq_dense.h"



void f_knnSeq_dense(float* data, int* G_Id, float* K, int* K_Id, int knn, int leaf_id, int pt_id, int leaf_size, int d){

	int start = leaf_id * leaf_size;


 
  float c;
  float norm;
  std::pair<float, int> pairt[leaf_size];
  int *Id_tmp;
  float *k_tmp;
  Id_tmp = (int *)malloc(sizeof(int) * leaf_size);
  k_tmp = (float *)malloc(sizeof(float) * leaf_size);
  
  //for (int i=0; i< leaf_size; i++){
    int i = pt_id;
    int perm_i = G_Id[leaf_id * leaf_size + pt_id];
    
    norm = 0.0;
    float norm_i = 0;
    //norm += V[ind0_i + i] * V[ind0_i + i];
    for (int s = 0; s < d; s++) norm_i += V[ind0_i + s] * V[ind0_i + s];
    for (int j=0; j < leaf_size; j++){
      int perm_j = G_Id[leaf_size * leaf_size + j];

      
      norm = norm_i;
      for (int s = 0; s < nnz_j; s++) norm += V[perm_j* d + s] * V[perm_j * d + s];

      c = 0;
      
      for (int n_i = 0; n_i < d; n_i++) c += V[perm_j * d + n_i] * V[perm_i * d + n_i];

      c = -2*c + norm; 
      c = (c>0) ? sqrt(c): 0.0;
    
      k_tmp[j] = c;
      Id_tmp[j] = G_Id[leaf_size * leaf_id + j];
     
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
