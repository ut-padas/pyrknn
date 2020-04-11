#include "denknn.hpp"
#include "orthogonal.hpp"
#include "gemm.hpp"
#include "reorder.hpp"
#include "merge.hpp"
#include "op_gpu.hpp"
#include "timer_gpu.hpp"
#include "util_gpu.hpp"
#include "print.hpp"


void build_tree(fvec&, int, int, ivec&, float&);

void leaf_knn(const ivec&, const fvec&, int, int, int, int*, float*, int, int, int);

int current_time_nanoseconds() {
    struct timespec tm;
    clock_gettime(CLOCK_REALTIME, &tm);
    return tm.tv_nsec;
}


void denknn(const int* hID, const float *hP, int n, int d, int level, int nTree,
    int *hNborID, float *hNborDist, int k, int blkPoint) {

  const int nLeaf = 1<<level;
  const int nPoint = (n+nLeaf-1)/nLeaf; // # points per leaf node
  const int nExtra = nPoint*nLeaf - n;
  const int N = n + nExtra;

  std::cout<<"\n========================"
           <<"\nPoints"
           <<"\n------------------------"
           <<"\n# points: "<<N
           <<"\n# dimensions: "<<d
           <<"\nmem: "<<n/1.e9*d*4<<" GB"
           <<"\n------------------------"
           <<"\n# artificial points: "<<nExtra
           <<"\n# points/leaf: "<<nPoint
           <<"\n# leaf nodes: "<<nLeaf
           <<"\n------------------------"
           <<"\nmem output: "<<n/1.e9*k*4*4<<" GB"
           <<"\nmem orthogonal bases: "<<d/1.e9*level*4<<" GB"
           <<"\nmem projection: "<<n/1.e9*level*4<<" GB"
           <<"\n========================\n"
           <<std::endl;


  // copy data to GPU
  dvec<int> dID(N);
  thrust::copy(hID, hID+n, dID.begin());  

  dvec<float> dP(N*d);
  thrust::copy(hP, hP+n*d, dP.begin()); // both in row-major

  dvec<int> dNborID(N*k*2, std::numeric_limits<int>::max());
  dvec<float> dNborDist(N*k*2, std::numeric_limits<float>::max());
  auto zero = thrust::make_counting_iterator<int>(0);
  auto iter = thrust::make_transform_iterator(zero, firstKCols(k, 2*k));
  auto leftKColsID = thrust::make_permutation_iterator(dNborID.begin(), iter);
  auto leftKColsDist = thrust::make_permutation_iterator(dNborDist.begin(), iter);
  thrust::copy(hNborID, hNborID+n*k, leftKColsID);
  thrust::copy(hNborDist, hNborDist+n*k, leftKColsDist);

  // insert artificial points at infinity
  thrust::sequence(dID.begin()+n, dID.end(), -nExtra, 1); // negative id
  thrust::fill(dP.begin()+n*d, dP.end(), -std::numeric_limits<float>::max());
  
  //tprint(N, d, dP, "Points on GPU");


  // local ordering
  dvec<int> order(N);
  thrust::sequence(order.begin(), order.end(), 0);

  // allocate memory for random arrays and projections
  dvec<float> R(d*level); // column-major
  dvec<float> Y(N*level); // column-major
      
  // -----------------------
  // timing
  // -----------------------
  float t_tree = 0., t_knn = 0., t_merge = 0.;
  float t_orth = 0., t_gemm = 0., t_sort = 0.;
  TimerGPU t, t1;
  
  // -----------------------
  // random seed
  // -----------------------
  int seed = current_time_nanoseconds();

  // -----------------------
  // Start KNN
  // -----------------------
  for (int tree=0; tree<nTree; tree++) {
  
    // cluster points
    {
      dvec<int> perm(N);
      t.start();
      // compute permutation
      {
        // generate random bases
        thrust::counting_iterator<int> start(tree*d*level);
        thrust::transform(start, start+d*level, R.begin(), prg(seed));

        t1.start();
        orthogonal(R, d, level);
        t1.stop(); t_orth += t1.elapsed_time();
        //tprint(level, d, R, "random projection");

        // gemm
        t1.start();
        GEMM(N, level, d, dP, R, Y);
        t1.stop(); t_gemm += t1.elapsed_time();
        std::cout<<"Finished GEMM.\n"<<std::endl;

        // compute permutation
        build_tree(Y, N, level, perm, t_sort);
      }
      t.stop(); t_tree += t.elapsed_time();
      //std::cout<<"Finished tree construction.\n"<<std::endl;

      // shuffle
      gather(dP, N, d, perm);
      gather(dID, perm);
      gather(order, perm);
    }

    // compute neighbors at leaf level
    t.start();
    int *curNborID = thrust::raw_pointer_cast(dNborID.data()+k);
    float *curNborDist = thrust::raw_pointer_cast(dNborDist.data()+k);    
    leaf_knn(dID, dP, nPoint, d, nLeaf, curNborID, curNborDist, k, 2*k, blkPoint);
    t.stop(); t_knn += t.elapsed_time();

    // shuffle results
    scatter(curNborID, N, 2*k, k, order);
    scatter(curNborDist, N, 2*k, k, order);

    t.start();
    std::cout<<"Merge neighbors\n";
    merge_neighbors(dNborDist, dNborID, n, 2*k, k);
    t.stop(); t_merge += t.elapsed_time();
  }

  std::cout<<"\n==========================="
           <<"\n    Dense KNN Timing"
           <<"\n---------------------------"
           <<"\n* Build tree: "<<t_tree<<" s"
           <<"\n\t- orthogonal: "<<t_orth<<" s"
           <<"\n\t- gemm: "<<t_gemm<<" s"
           <<"\n\t- sort: "<<t_sort<<" s"
           <<"\n* Leaf KNN: "<<t_knn<<" s"
           <<"\n* Merge: "<<t_merge<<" s"
           <<"\n===========================\n"
           <<std::endl;

  // -----------------------
  // Copy results back to CPU
  // -----------------------
  thrust::copy(leftKColsID, leftKColsID+n*k, hNborID);
  thrust::copy(leftKColsDist, leftKColsDist+n*k, hNborDist);
}


