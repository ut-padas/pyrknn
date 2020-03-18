CC = nvcc
LFLAG=-lcublas -lcurand

all: sgmv_timer sgemm_timer

sgemm_timer: timer.o sgemm.o
	$(CC) -o $@ $^ $(LFLAG)

sgmv_timer: timer.o sgmv.o
	$(CC) -o $@ $^ $(LFLAG)

timer.o: timer.cpp timer.hpp
	$(CC) -c $< -o $@  

sgmv.o: sgmv.cu
	$(CC) -c $^ -o $@ $(LFLAG)

sgemm.o: sgemm.cu
	$(CC) -c $^ -o $@ $(LFLAG)

.PHONY: clean
clean:
	rm -f *.o sgmv_timer