#ifndef MATRIX_HPP
#define MATRIX_HPP


class SpMat {
public:
  SpMat() {}

  SpMat(int m_, int n_, int nnz_, int *row_, int *col_, float *val_, int node_)
    : m(m_), n(n_), nnz(nnz_), rowPtr(row_), colIdx(col_), val(val_), nNodes(node_) {}

  int rows() const {return m;}
  int cols() const {return n;}

public:
  int m, n, nnz;
  int *rowPtr, *colIdx;
  float *val;
  int nNodes;
};


#endif
