
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>

void gen_col_data(int M, int d, int *R, int *C, float *V);
void gen_row(int M, int nnzperrow, int *R, int *G_ID, int d, bool var_nnz);

void gen_rnd_dense(float *data, int *G_Id, int M, int d);
