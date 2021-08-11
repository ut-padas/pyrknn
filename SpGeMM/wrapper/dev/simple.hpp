#ifndef SIMPLE_HPP
#define SIMPLE_HPP

#include<stdio.h>
#include <stdlib.h>
//#include <cusparse.h>
//#include <cuda.h>
//#include <helper_cuda.h>
//#include <cuda_runtime.h>
#include <algorithm>



void gen_sparse(int M, int tot_nnz, int d, int *R, int *C, float *V);

void gen_R(int M, int nnzperrow, int *R, int *G_Id, int d);


void gpu_knn(int *R, int *C, float *V, int *G_Id, int M, int leaves, int k, float *knn, int *knn_Id, int max_nnz);


#endif //SIMPLE_HPP
