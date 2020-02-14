# distutils: language = c++

import numpy as np
cimport numpy as np

from combinatorics cimport Scan, sampleWithoutReplacement, Select, Accumulate, Reduce
from libcpp.vector cimport vector
from cython.operator cimport dereference as deref

# cdef int n = 10
# cdef vector[int] vect = vector[int](n,1)

# cdef vector[int] result = sampleWithoutReplacement(8,vect)
# print(result.size())

# cdef vector[int] *vect = new vector[int](n,1)
# for i in range(n):
#     deref(vect)[i] = 1

# cdef vector[int] result = Scan[int,int](deref(vect))
# print(result)
# print(result.size())

# cdef vector[int] vect
# cdef int i, x

# for i in range(10):
#     vect.push_back(i)

# for i in range(10):
#     print(deref(vect)[i])

# for x in (vect):
#     print(x)

def scan(l):
    if type(l) is not np.ndarray:
        raise TypeError("The passed argument is not a numpy array.")
    if l.dtype == np.dtype('int32'):
        return Scan[int,int](l)
    elif l.dtype == np.dtype('float32'):
        return Scan[float,float](l)
    elif l.dtype == np.dtype('float64'):
        return Scan[double,double](l)
    else:
        raise TypeError("Unsupported data element type.")

def sample_without_replacement(l,int n):
    if type(l) is not np.ndarray:
        raise TypeError("The passed argument is not a numpy array.")
    if l.dtype == np.dtype('int32'):
        return sampleWithoutReplacement[int](n,l)
    elif l.dtype == np.dtype('float32'):
        return sampleWithoutReplacement[float](n,l)
    elif l.dtype == np.dtype('float64'):
        return sampleWithoutReplacement[double](n,l)

def accumulate(l, i):
    if type(l) is not np.ndarray:
        raise TypeError("The passed argument is not a numpy array.")
    if l.dtype == np.dtype('int32'):
        return Accumulate[int](l,i)
    elif l.dtype == np.dtype('float32'):
        return Accumulate[float](l,i)
    elif l.dtype == np.dtype('float64'):
        return Accumulate[double](l,i)

def reduce_par(l, i):
    if type(l) is not np.ndarray:
        raise TypeError("The passed argument is not a numpy array.")
    if l.dtype == np.dtype('int32'):
        return Reduce[int](l,i)
    elif l.dtype == np.dtype('float32'):
        return Reduce[float](l,i)
    elif l.dtype == np.dtype('float64'):
        return Reduce[double](l,i)


def select(l, size_t k):
    if type(l) is not np.ndarray:
        raise TypeError("The passed argument is not a numpy array.")
    if l.dtype == np.dtype('int32'):
        return Select[int](k,l)
    elif l.dtype == np.dtype('float32'):
        return Select[float](k,l)
    elif l.dtype == np.dtype('float64'):
        return Select[double](k,l)
