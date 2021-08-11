

default: pyexamples

pyexamples: setup.py test.pyx libtest.a
	python setup.py build_ext --inplace && rm -f test.c && rm -Rf build

 
libtest.a : test.o
	ar rcs $@ $^

test.o: test.cpp
	g++ -c $<

clean:
	rm *.o *.a



