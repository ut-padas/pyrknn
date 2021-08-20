#ifndef timer_hpp
#define timer_hpp

#include <string>

class Timer {
public:
  void start();
  void stop();
  double elapsed_time();
  void show_elapsed_time();
  void show_elapsed_time(const char*);
  std::string get_elapsed_time(const char*);

private:
  double tStart = 0.0;
  double tStop = 0.0;
};

#endif /* timer_hpp */
