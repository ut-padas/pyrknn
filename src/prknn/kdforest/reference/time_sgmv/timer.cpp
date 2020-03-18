#include "timer.hpp"

#include <sys/time.h>
#include <sstream>
#include <iostream>

// Return time in seconds since the Unix epoch
static double timer() {
  double time;
  struct timeval tv;
  gettimeofday(&tv, NULL);
  time = (double)tv.tv_sec + (double)tv.tv_usec/1.e6;
  return time;
}

void Timer::start() {
  tStart = timer();
}

void Timer::stop() {
  tStop = timer();
}

double Timer::elapsed_time() {
  return tStop-tStart;
}

void Timer::show_elapsed_time() {
  std::cout << "Elapsed time : " << tStop-tStart << ".\n";
}

void Timer::show_elapsed_time(const char* msg) {
  std::cout << msg << " : "
	    << tStop-tStart << " seconds." << std::endl;
}

std::string Timer::get_elapsed_time(const char* msg) {
  std::stringstream ss;
  ss << msg << " : " << tStop-tStart << " seconds." << std::endl;
  return ss.str();
}