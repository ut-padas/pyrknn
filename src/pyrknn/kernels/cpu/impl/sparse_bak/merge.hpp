#ifndef merge_hpp
#define merge_hpp

#include <omp.h>
#include <utility>

using namespace std;

template<typename key_t, typename value_t>
inline bool less_key(const pair<key_t, value_t> &a, const pair<key_t, value_t> &b ){
    return (a.first < b.first);
}

template<typename key_t, typename value_t>
inline bool less_value(const pair<key_t, value_t> &a, const pair<key_t, value_t> &b){
    return (a.second < b.second);
}

template<typename key_t, typename value_t>
inline bool equal_key(const pair<key_t, value_t> &a, const pair<key_t, value_t> &b ){
    return (a.first == b.first);
}

template<typename key_t, typename value_t>
inline bool equal_value(const pair<key_t, value_t> &a, const pair<key_t, value_t> &b){
    return (a.second == b.second);
}


template<typename T>
void merge_neighbor_cpu(T* D1, unsigned* I1, const T* D2, const unsigned* I2, 
    unsigned n, unsigned k, int cores){

    omp_set_num_threads(cores);

    #pragma omp parallel
    {

        std::vector<std::pair<unsigned, T>> aux(2*k);

        #pragma omp for
        for(size_t i = 0; i < n; ++i){
            size_t offset = i * k;
            //Merge twolists
            for(size_t j = 0; j < k; ++j){
                aux[ j]   = std::make_pair(I1[offset + j], D1[offset + j]);
                aux[ k+j] = std::make_pair(I2[offset + j], D2[offset + j]);
            }

            //Sort according to index
            std::sort(aux.begin(), aux.end(), less_key<unsigned, T>);
            auto last = std::unique(aux.begin(), aux.end(), equal_key<unsigned, T> );
            //std::sort(aux.begin(), last, less_value<unsigned, T>);
            std::nth_element(aux.begin(), aux.begin()+k-1, last, less_value<unsigned, T>);

            //Copy Results back to output
            for(size_t j = 0; j < k; ++j){
                auto current = aux[j];
                I1[offset+j] = current.first;
                D1[offset+j] = current.second;
            }
        }
        
    }
    
}

#endif
