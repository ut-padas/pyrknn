import cython

cdef extern from "test.hpp":
  void hello(const char *name)

def py_hello(name: bytes) -> None:
  hello(name)
