#ifndef KNN_GPU_HPP
#define KNN_GPU_HPP

void gemm_kselect_opt(int, float*[], float*[], int*[], int, int, float*[], int*[], int, int,
                    float&, float&, float&);


void knn_gpu(float* ptrR[], float *ptrQ[], int *ptrID[], float *ptrNborDist[], int *ptrNborID[], int nLeaf, int N, int d, int k, int m
#ifdef PROD
    , int device);
#else
    , float &t_dist, float &t_sort, float &t_kernel);
#endif


#endif //KNN_GPU_HPP
