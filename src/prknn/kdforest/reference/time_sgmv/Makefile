CC = nvcc
LFLAG=-lcublas -lcurand

all: sgmv_timer

sgmv_timer: timer.o sgmv.o
	$(CC) -o $@ $^ $(LFLAG)

timer.o: timer.cpp timer.hpp
	$(CC) -c $< -o $@  

sgmv.o: sgmv.cu
	$(CC) -c $^ -o $@ $(LFLAG)

.PHONY: clean
clean:
	rm -f *.o sgmv_timer