
EXE=driver
LIB=libreadSVM.so

DEBUG?=0

default: $(LIB) $(EXE)

CFLAG=-Wall -I $(EIGEN_ROOT)
LFLAG=-L. -lreadSVM

ifeq ($(DEBUG), 1)
  $(info ----------------------)
  $(info *** debugging mode ***)
  $(info ----------------------)
	CFLAG+=-g
else
	CFLAG+=-O3
endif

KERNEL=readSVM.o
OBJS= driver.o

readSVM.o: readSVM.cpp
	g++ -c $< $(CFLAG) -shared -fPIC -I ../util

driver.o: driver.cpp
	g++ -c $< $(CFLAG) -I ../util

$(LIB): $(KERNEL)
	g++ -shared $^ -o $@

$(EXE): $(OBJS) $(LIB)
	g++ $< ../util/timer.o -o $@ $(LFLAG)

run: $(EXE)
	$(EXE)

clean:
	rm -rf *.o *.so $(EXE)


