#ifndef SINGLETON_HPP
#define SINGLETON_HPP

#include <memory>

template<typename T>
class Singleton {
  friend class knnHandle_t; // access private constructor/destructor
  friend class mgpuHandle_t;
public:
  static T& instance() {
    static const std::unique_ptr<T> instance{new T()};
    return *instance;
  }
private:
  Singleton() {};
  ~Singleton() {};
  Singleton(const Singleton&) = delete;
  void operator=(const Singleton&) = delete;
};

#endif
