#ifndef nla_hpp
#define nla_hpp

#include "matrix.hpp"

// numerical linear algebra routines

void orthonormalize(fMatrix&);

void GEMM_SDD(Points &P, fMatrix &R, fMatrix &X);
  
void GEMM_SDD(unsigned m, unsigned n, unsigned k, int *rowPtr, int *colIdx, float *val, 
    unsigned nnz, float *R, float *X);


void gather(Points &P, const ivec &perm, dvec&);


void scatter(Points &P, const ivec &perm, dvec&);


void compute_row_norm(const Points &, fvec&, double&);


void compute_distance(const Points &P, const fvec&, fMatrix &Dt, dvec&);


void inner_product(const Points&, const Points&, float*);


namespace par {
  void copy(unsigned, float*, float*);
}


/*
void orthonormal_bases(float*, unsigned, unsigned);

*/

#endif
