#include "util.hpp"

#include <time.h>

int current_time_nanoseconds(){
  struct timespec tm;
  clock_gettime(CLOCK_REALTIME, &tm);
  return tm.tv_nsec;
}



