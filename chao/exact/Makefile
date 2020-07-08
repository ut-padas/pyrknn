
CC=mpicxx

EXE=driver
LIB=libexact.so

DEBUG?=0

default: $(LIB) $(EXE)

CFLAG=-Wall -I $(EIGEN_ROOT) -std=c++11
LFLAG=-L. -lexact

ifeq ($(DEBUG), 1)
  $(info ----------------------)
  $(info *** debugging mode ***)
  $(info ----------------------)
	CFLAG+=-g
else
	CFLAG+=-O3
endif

KERNEL=exact.o
OBJS= driver.o

exact.o: exact.cpp
	$(CC) -c $< $(CFLAG) -shared -fPIC

driver.o: driver.cpp
	$(CC) -c $< $(CFLAG)

$(LIB): $(KERNEL)
	$(CC) -shared $^ -o $@

$(EXE): $(OBJS) $(LIB)
	$(CC) $(OBJS) -o $@ $(LFLAG)

run: $(EXE)
	ibrun -np 16 $(EXE) -d 64 -q 100 -r 90 -k 16

clean:
	rm -rf *.o *.so $(EXE)


