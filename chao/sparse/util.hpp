#ifndef UTIL_HPP
#define UTIL_HPP

#include <string>

void print(int m, int n, int nnz, int *rowPtr, int *colIdx, float *val, const std::string &);

void print(int m, int n, float *val, const std::string &);

void dprint(int m, int n, int nnz, int *rowPtr, int *colIdx, float *val, const std::string &);

void dprint(int, int*, const std::string&);
void dprint(int, float*, const std::string&);

void dprint(int, int, int*, const std::string&);
void dprint(int, int, float*, const std::string&);

void copy_spmat_d2h(int, int, int, int*, int*, float*, int*, int*, float*);

void copy(int, float*, float*);

void copy(int, int*, int*);

#endif
