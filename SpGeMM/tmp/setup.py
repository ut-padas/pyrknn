from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

examples_extension = Extension(name="test",
  sources=["test.pyx"],
  language='c++'
)

setup(
  name="test",
  ext_modules=cythonize([examples_extension])
)


