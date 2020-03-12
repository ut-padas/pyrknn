#ifndef KNN_GPU_HPP
#define KNN_GPU_HPP

void gemm_kselect_opt(int, float*[], float*[], int*[], int, int, float*[], int*[], int, int,
                    float&, float&, float&);

#endif
