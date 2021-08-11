
#include "rnd_sparse.h"

void gen_col_data(int M, int d, int *R, int *C, float *V) {

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


void gen_row(int M, int nnzperrow, int *R, int *G_Id, int d, bool var_nnz = true) {

  R[0] = 0;
  int tot_nnz = 0;
  int val;
  for (int m =1; m <= M; m++){
    val = (var_nnz) ? 1 + rand()%(2*nnzperrow) : nnzperrow;
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


void gen_rnd_dense(float *data, int M, int d){

  for (int i=0; i < M; i++){
    for (int j = 0; j < d; j++){
    int ind = i * d + j
    data[ind] = ((float) rand()) / (float) RAND_MAX;
    }
  }
}

