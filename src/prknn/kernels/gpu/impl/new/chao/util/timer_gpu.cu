#include "timer_gpu.hpp"

#include <sys/time.h>
#include <sstream>
#include <iostream>

//#include 

// Return time in seconds since the Unix epoch
static double timer() {
  double time;
  struct timeval tv;
  gettimeofday(&tv, NULL);
  time = (double)tv.tv_sec + (double)tv.tv_usec/1.e6;
  return time;
}

void TimerGPU::start() {
  cudaDeviceSynchronize();
  tStart = timer();
}

void TimerGPU::stop() {
  cudaDeviceSynchronize();
  tStop = timer();
}

double TimerGPU::elapsed_time() {
  return tStop-tStart;
}

void TimerGPU::show_elapsed_time() {
  std::cout << "Elapsed time : " << tStop-tStart << ".\n";
}

void TimerGPU::show_elapsed_time(const char* msg) {
  std::cout << msg << " : "
	    << tStop-tStart << " seconds." << std::endl;
}

std::string TimerGPU::get_elapsed_time(const char* msg) {
  std::stringstream ss;
  ss << msg << " : " << tStop-tStart << " seconds." << std::endl;
  return ss.str();
}
