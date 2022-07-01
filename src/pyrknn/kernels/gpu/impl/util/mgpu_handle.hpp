#ifndef MGPU_HANDLE_HPP
#define MGPU_HANDLE_HPP

#include <iostream>
#include "singleton.hpp"
#include <moderngpu/kernel_segsort.hxx>


class mgpuHandle_t final: public Singleton<mgpuHandle_t>{
  friend class Singleton<mgpuHandle_t>; // access private constructor/destructor
private:
  mgpuHandle_t() {
    //std::cout<<"Create mgpuHandle_t instance"<<std::endl;
    // mgpu context
    ctx = new mgpu::standard_context_t(false);
  }
public:
  mgpu::standard_context_t& mgpu_ctx() {return *ctx;}
  ~mgpuHandle_t() {
    //std::cout<<"Destroy mgpuHandle_t instance"<<std::endl;
    delete ctx;
  }
public:
  mgpu::standard_context_t *ctx;
};

#endif
