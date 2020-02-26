#ifndef KNN_HPP
#define KNN_HPP

void bb_gemm_kselect(int, float*[], float*[], int*[], int, int, float*[], int*[], int, 
                bool debug=false);

void distSquared_gpu_stream(int, float*[], float*[], float*[], int*, int*, int);

void kselect_gpu_stream(int, float*[], int*[], int*, int*, float*[], int*[], int);

void distSquared_gpu(const float*, const float*, float*, int, int, int);

void kselect_gpu(const float*, const int*, int, float *, int *, int);

void merge_neighbor_gpu(const float*, const int*, int,
		    const float*, const int*, int,
		    float*, int*, int);

#endif
