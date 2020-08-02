
EXE=driver
LIB=libspknncpu.so
OMP=OMP_NUM_THREADS=112 KMP_AFFINITY=compact

DEPDIR=../../../gpu/impl/readSVM
DEP=../../../gpu/impl/readSVM/libreadSVM.so 


CC?=g++
DEBUG?=0
BACKEND?=backend_eigen.cpp

CFLAG=-Wall -I $(EIGEN_ROOT) -std=c++11 -shared -fPIC 
LFLAG=-L./ -lspknncpu -L$(DEPDIR) -lreadSVM

ifeq ($(DEBUG), 1)
  $(info ----------------------)
  $(info *** debugging mode ***)
  $(info ----------------------)
	CFLAG+=-g 
else
	CFLAG+=-O3
endif


#ifeq ($(MKL), 1)
#  $(info ----------------------)
#  $(info ***  Linking MKL   ***)
#  $(info ----------------------)
#	CFLAG+=-DEIGEN_USE_MKL_ALL -mkl
#	LFLAG+=-mkl
#endif


ifeq ($(findstring frontera,$(shell uname -n)),frontera)
  $(info -------------------------)
  $(info ***  Use MKL backend  ***)
  $(info -------------------------)
	CC=icc
	CFLAG+=-mkl -qopenmp #-xCORE-AVX512
	LFLAG+=-mkl -qopenmp
	BACKEND=backend_mkl.cpp
else
  $(info ---------------------------)
  $(info ***  Use Eigen backend  ***)
  $(info ---------------------------)
	CFLAG+=-fopenmp
	LFLAG+=-fopenmp
endif

KERNEL=spknn.o backend.o util.o timer.o
OBJS=driver.o timer.o util_eigen.o


default: $(EXE)

$(EXE): $(OBJS) $(LIB) $(DEP) 
	$(CC) $(OBJS) $(LIB) -o $@ $(LFLAG)
	
$(LIB): $(KERNEL)
	$(CC) -shared $^ -o $@
		
driver.o: driver.cpp
	$(CC) -c $< $(CFLAG) -I $(DEPDIR)

spknn.o: spknn.cpp spknn.hpp matrix.hpp merge.hpp
	$(CC) -c $< $(CFLAG)

backend.o: $(BACKEND) NLA.hpp
	$(CC) -c $< $(CFLAG) -o $@

%.o: %.cpp %.hpp
	$(CC) -c $< $(CFLAG)

$(DEP):
	make -C $(DEPDIR) -j8

run: $(EXE)
	$(EXE) 

debug: $(EXE)
	$(EXE) -n 12 -d 5 -l 2 -k 3 -bt 1 -t 1

benchmark: $(EXE)
	#$(EXE) -dataset avazu_app_t -l 10 -k 5 -t 3 -bl 512 -bp 256 -bt 1
	#$(EXE) -dataset avazu_app -l 13 -k 16 -t 200 -bl 512 -bp 256 -bt 1
	#time $(EXE) -dataset avazu -l 17 -k 5 -t 200 -bl 1024 -bp 256 -bt 1
	#$(PROF) $(EXE) -dataset avazu -l 15 -k 5 -t 2 -bl 1024 -bp 64 -bt 1
	#$(EXE) -n 524288 -d 10000 -sparse 0.0001 -k 64 -t 1 -bl 10 -bp 64 -bt 1 -l 8
	#$(EXE) -dataset test -k 64 -t 1 -bl 512 -bp 64 -bt 1 -l 8
	#
	#$(EXE) -dataset url -l 12 -k 64 -t 3 -bl 512 -bp 300 -bt 1
	#$(EXE) -dataset avazu -l 18 -k 5 -t 200 -bl 1024 -bp 256 -bt 1
	#$(EXE) -dataset kdd -k 8
	#$(EXE) -dataset criteo -k 8
	#$(OMP) $(EXE) -dataset url -k 64 -l 10 -bt 1 -t 1
	#$(OMP) $(EXE) -dataset avazu -l 16 -k 64 -bt 1 -t 3
	#$(OMP) $(EXE) -dataset criteo -k 64 -l 15 -bt 1 -t 1
	$(OMP) $(EXE) -dataset kdd -k 64 -l 13 -bt 1 -t 3

clean:
	rm -f *.o $(EXE) $(LIB)

