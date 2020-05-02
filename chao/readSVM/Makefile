
EXE=driver
LIB=libreadSVM_new.so

DEBUG?=0

default: $(LIB) $(EXE)

CFLAG=-Wall -I $(EIGEN_ROOT) -std=c++11
LFLAG=-L. -lreadSVM_new

ifeq ($(DEBUG), 1)
  $(info ----------------------)
  $(info *** debugging mode ***)
  $(info ----------------------)
	CFLAG+=-g
else
	CFLAG+=-O3
endif

KERNEL=readSVM.o
OBJS= driver.o timer.o

readSVM.o: readSVM.cpp
	g++ -c $< $(CFLAG) -shared -fPIC

timer.o: timer.cpp
	g++ -c $< $(CFLAG)

driver.o: driver.cpp
	g++ -c $< $(CFLAG)

$(LIB): $(KERNEL)
	g++ -shared $^ -o $@

$(EXE): $(OBJS) $(LIB)
	g++ $(OBJS) -o $@ $(LFLAG)

run: $(EXE)
	$(EXE)

clean:
	rm -rf *.o *.so $(EXE)


