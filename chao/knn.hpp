#ifndef KNN_HPP
#define KNN_HPP

void gemm_kselect_opt(int, float*[], float*[], int*[], int, int, float*[], int*[], int, 
                float&, float&, float&, float&);

void bb_gemm_kselect(int, float*[], float*[], int*[], int, int, float*[], int*[], int, 
                float&, float&, float&, float&);

void distSquared_gpu_stream(int, float*[], float*[], float*[], int*, int*, int);

void kselect_gpu_stream(int, float*[], int*[], int*, int*, float*[], int*[], int);

void distSquared_gpu(const float*, const float*, float*, int, int, int);

void kselect_gpu(const float*, const int*, int, float *, int *, int);

void merge_neighbor_gpu(const float*, const int*, int,
		    const float*, const int*, int,
		    float*, int*, int);

template<typename T>
void print(const std::vector<T>& vec, const std::string &name) {
  std::cout<<std::endl<<name<<":"<<std::endl;
  for (size_t i=0; i<vec.size(); i++)
    std::cout<<vec[i]<<" ";
  std::cout<<std::endl<<std::endl;
}


#endif
