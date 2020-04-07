#include "merge.hpp"
#include "sort.hpp"
#include "op_gpu.hpp"
#include "print.hpp"

#include <iomanip>
#include <thrust/gather.h>

struct strideBlk: public thrust::unary_function<int, int> {

  int start; 
  int iCols;
  int oCols;

  __host__ __device__
  strideBlk(int s, int in, int out): start(s), iCols(in), oCols(out) {}
  
  __host__ __device__
  int operator()(int i) {
    return start + i/iCols*oCols + i%iCols;
  }
};


struct sameNborID: public thrust::binary_function<int, int, bool> {

  int nCols;
  int *ID;

  __host__ __device__
  sameNborID(int c, int *id): nCols(c), ID(id) {}

  __host__ __device__
  bool operator()(int i, int j) {
    return i/nCols==j/nCols && ID[i]==ID[j];
  }
};


struct firstKColumn: public thrust::unary_function<int, int> {

  int k;
  const int *perm, *head;

  __host__ __device__
  firstKColumn(int k_, int *p, int *h): k(k_), perm(p), head(h) {}
    
  __host__ __device__
  int operator()(int i){
    return perm[ head[i/k] + i%k ]; 
  }
};


struct firstKIdx: public thrust::unary_function<int, int> {

  int k;
  const int *head;

  __host__ __device__
  firstKIdx(int k_, int *h): k(k_), head(h) {}
    
  __host__ __device__
  int operator()(int i){
    return head[i/k] + i%k; 
  }
};


struct rowIdx: public thrust::unary_function<int, int> {
  const int nCols;
  
  __host__ __device__
    rowIdx(int n): nCols(n) {}

  __host__ __device__
    int operator()(const int i) const {
      return i/nCols;
    }
};


template <typename T>
void free(dvec<T> &x) {
  x.clear();
  x.shrink_to_fit();
}


void merge_neighbors_gpu(float *hDist, int *hID, int m, int n, int k) {
  fvec dDist(hDist, hDist+m*n);
  ivec dID(hID, hID+m*n);
  merge_neighbors(dDist, dID, m, n, k);
  thrust::copy(dDist.begin(), dDist.end(), hDist);
  thrust::copy(dID.begin(), dID.end(), hID);
}


// m x n matrix -> m x k matrix
// Notice the two matrices may have more than 'm' rows
void merge_neighbors(dvec<float> &nborDist, dvec<int> &nborID, int m, int n, int k) {
  
  // sort and unique IDs
  dvec<int> idx(m*n);
  sort_matrix_rows_mgpu(nborID, idx, m, n);
  //std::cout<<"Finished sort"<<std::endl;


  int *ID = thrust::raw_pointer_cast(nborID.data());
  dvec<int> count(m*n);
  thrust::sequence(count.begin(), count.end(), 0);
  auto end = thrust::unique_by_key(count.begin(), count.end(), idx.begin(), sameNborID(n, ID));
  idx.erase(end.second, idx.end());
  count.erase(end.first, count.end());
  //std::cout<<"Finished unique, unique size: "<<idx.size()<<std::endl;


  // get (unique) distance and ID
  dvec<float> uniqueDist(idx.size());
  dvec<int> uniqueID(idx.size());
  thrust::gather(idx.begin(), idx.end(), nborDist.begin(), uniqueDist.begin());
  thrust::gather(count.begin(), count.end(), nborID.begin(), uniqueID.begin());
  free(count);
  //std::cout<<"Finished copying unique"<<std::endl;

    
  // get segment length
  dvec<int> segments(m+1, 0);
  auto rowIter = thrust::make_transform_iterator(idx.begin(), rowIdx(n));
  auto zero = thrust::make_counting_iterator<int>(0);
  thrust::upper_bound(rowIter, rowIter+idx.size(), zero, zero+m, segments.begin()+1);
  //std::cout<<"Finished segment"<<std::endl;


  sort_matrix_rows_mgpu(uniqueDist, idx, idx.size(), segments, m);
  //std::cout<<"Finished sort"<<std::endl;
    
  
  // get first k-cols
  ID = thrust::raw_pointer_cast(idx.data());
  int *head  = thrust::raw_pointer_cast(segments.data());

  auto iterI = thrust::make_transform_iterator(zero, firstKColumn(k, ID, head));
  auto permI = thrust::make_permutation_iterator(uniqueID.begin(), iterI);

  auto iterD = thrust::make_transform_iterator(zero, firstKIdx(k, head));
  auto permD = thrust::make_permutation_iterator(uniqueDist.begin(), iterD);


  auto it = thrust::make_transform_iterator(zero, firstKCols(k, n));
  auto PI = thrust::make_permutation_iterator(nborID.begin(), it);
  auto PD = thrust::make_permutation_iterator(nborDist.begin(), it);

  thrust::copy(thrust::device, permI, permI+m*k, PI);
  thrust::copy(thrust::device, permD, permD+m*k, PD);  
  //std::cout<<"Finished merge"<<std::endl;
}


void merge_neighbors(float *nborD1, int *nborI1, const float *nborD2, const int *nborI2,
    int m, int n, int k) {
    
#ifndef PROD
  float t_copy, t_sort1, t_unique, t_copy2;
  float t_seg, t_sort2, t_out, t_kernel;
  bool debug;
  TimerGPU t, tk; 
  tk.start();
  t.start();
#endif

  std::cout<<"[Merge] Allocate memory "<<m/1.e9*n*2*4<<" GB"<<std::endl;
  dvec<int> nborID(2*n*m);
  dvec<float> nborDist(2*n*m);

  // TODO: copy memory in parallel?
  // copy the first half
  auto zero  = thrust::make_counting_iterator<int>(0);
  auto iter1 = thrust::make_transform_iterator(zero, strideBlk(0, n, 2*n));
  auto permI1 = thrust::make_permutation_iterator(nborID.begin(), iter1);
  auto permD1 = thrust::make_permutation_iterator(nborDist.begin(), iter1);
  thrust::copy(thrust::device, nborI1, nborI1+m*n, permI1);
  thrust::copy(thrust::device, nborD1, nborD1+m*n, permD1);
  // the second half
  auto iter2 = thrust::make_transform_iterator(zero, strideBlk(n, n, 2*n));
  auto permI2 = thrust::make_permutation_iterator(nborID.begin(), iter2);
  auto permD2 = thrust::make_permutation_iterator(nborDist.begin(), iter2);
  thrust::copy(thrust::device, nborI2, nborI2+m*n, permI2);
  thrust::copy(thrust::device, nborD2, nborD2+m*n, permD2);

#ifndef PROD
  t.stop(); t_copy = t.elapsed_time();
  if (debug) {
    print(nborID, m, 2*n, "combined ID");
    print(nborDist, m, 2*n, "combined Dist");
  }
  t.start();
#endif
  
  // sort and unique IDs
  dvec<int> idx(nborID.size());
  auto IDcpy = nborID; // TODO: avoid memory copy
  sort_matrix_rows_mgpu(IDcpy, idx, m, 2*n);

#ifndef PROD
  t.stop(); t_sort1 = t.elapsed_time();
  t.start();
#endif

  int *ID = thrust::raw_pointer_cast(IDcpy.data());
  dvec<int> count(nborID.size());
  thrust::sequence(count.begin(), count.end(), 0);
  auto end = thrust::unique_by_key(count.begin(), count.end(), idx.begin(), sameNborID(2*n, ID));
  idx.erase(end.second, idx.end());

  // free resource
  count.clear();
  count.shrink_to_fit();

#ifndef PROD
  t.stop(); t_unique = t.elapsed_time();
  if (debug)
    print(idx, m, idx.size()/m, "idx");
  t.start();
#endif


  // get (unique) distance and ID
  dvec<float> uniqueDist(idx.size());
  dvec<int> uniqueID(idx.size());
  auto permD = thrust::make_permutation_iterator(nborDist.begin(), idx.begin());
  auto permI = thrust::make_permutation_iterator(nborID.begin(), idx.begin());
  thrust::copy(permD, permD+idx.size(), uniqueDist.begin());
  thrust::copy(permI, permI+idx.size(), uniqueID.begin());

  // free resource
  nborID.clear();
  nborDist.clear();
  nborID.shrink_to_fit();
  nborDist.shrink_to_fit();


#ifndef PROD
  t.stop(); t_copy2 = t.elapsed_time();  
  if (debug) {
    print(uniqueID, m, idx.size()/m, "unique ID");
    print(uniqueDist, m, idx.size()/m, "unique Dist");
  }
  t.start();
#endif


  // get segment length
  dvec<int> segments(m+1, 0);
  auto rowIter = thrust::make_transform_iterator(idx.begin(), rowIdx(2*n));
  thrust::upper_bound(rowIter, rowIter+idx.size(), zero, zero+m, segments.begin()+1);

#ifndef PROD
  t.stop(); t_seg = t.elapsed_time();
  if (debug) {
    print(segments, "segments");
  }
  t.start();
#endif

  // sort distance
  auto distCpy = uniqueDist;
  sort_matrix_rows_mgpu(distCpy, idx, idx.size(), segments, m);
    
#ifndef PROD
  t.stop(); t_sort2 = t.elapsed_time();
  
  if (debug) {
    print(distCpy, m, idx.size()/m, "distance");
    print(idx, m, idx.size()/m, "idx");
  }
  t.start();
#endif
  
  // get first k-cols
  ID = thrust::raw_pointer_cast(idx.data());
  int *head  = thrust::raw_pointer_cast(segments.data());
  auto iter  = thrust::make_transform_iterator(zero, firstKColumn(k, ID, head));
  auto permID = thrust::make_permutation_iterator(uniqueID.begin(), iter);
  auto permDist = thrust::make_permutation_iterator(uniqueDist.begin(), iter);

  //std::cout<<"Iter:"<<std::endl;
  //thrust::for_each(thrust::device, iter, iter+m*k, printf_functor());


  thrust::copy(thrust::device, permID, permID+m*k, nborI1);
  thrust::copy(thrust::device, permDist, permDist+m*k, nborD1);

#ifndef PROD
  t.stop(); t_out = t.elapsed_time();
  tk.stop(); t_kernel += tk.elapsed_time();
#endif
}


void merge_neighbors_gpu(float *nborD1, const float *nborD2, int *nborI1, const int *nborI2,
    int m, int n, float *ptrDist, int *ptrID, int k, 
    float &t_kernel, bool debug) {
  
  dvec<float> d_D1(nborD1, nborD1+m*n);
  dvec<float> d_D2(nborD2, nborD2+m*n);
  dvec<int> d_I1(nborI1, nborI1+m*n);
  dvec<int> d_I2(nborI2, nborI2+m*n);

  float t_copy = 0., t_sort1 = 0., t_unique = 0., t_copy2 = 0., 
        t_seg = 0., t_sort2 = 0., t_out = 0.;

  merge_neighbors(
      thrust::raw_pointer_cast(d_D1.data()), 
      thrust::raw_pointer_cast(d_I1.data()), 
      thrust::raw_pointer_cast(d_D2.data()), 
      thrust::raw_pointer_cast(d_I2.data()), 
      m, n, k
#ifdef PROD
      );
#else
      , t_copy, t_sort1, t_unique, t_copy2, t_seg, t_sort2, t_out, t_kernel, debug);
#endif

  // finalize
  thrust::copy(d_I1.begin(), d_I1.end(), ptrID);
  thrust::copy(d_D1.begin(), d_D1.end(), ptrDist);

  if (!debug) {
  std::cout<<"\n==============\n"
           <<"Kernel profile\n"
           <<"--------------\n"
           <<std::setprecision(2)
           <<"copy1: "<<t_copy/t_kernel*100<<" %\n"
           <<"sort1: "<<t_sort1/t_kernel*100<<" %\n"
           <<"unique: "<<t_unique/t_kernel*100<<" %\n"
           <<"copy2: "<<t_copy2/t_kernel*100<<" %\n"
           <<"segment: "<<t_seg/t_kernel*100<<" %\n"
           <<"sort2: "<<t_sort2/t_kernel*100<<" %\n"
           <<"out: "<<t_out/t_kernel*100<<" %\n"
           <<"==============\n\n";
  }
}

