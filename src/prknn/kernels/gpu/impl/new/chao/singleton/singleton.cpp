#include <iostream>
#include <memory>

template<typename T>
class Singleton {
public:
    static T& instance();

    Singleton(const Singleton&) = delete;
    Singleton& operator= (const Singleton) = delete;

protected:
    Singleton() {}
};

template<typename T>
T& Singleton<T>::instance()
{
    static const std::unique_ptr<T> instance{new T()};
    return *instance;
}


class Test final : public Singleton<Test>
{
public:
    Test() { std::cout << "constructed" << std::endl; }
    ~Test() {  std::cout << "destructed" << std::endl; }

    void use() const { std::cout << "in use" << std::endl; };
};

void create_level() {
    auto const& t = Test::instance();
    t.use();
}

void leaf_kernel() {
    auto const& t = Test::instance();
    t.use();
}

int main()
{
    // Test cannot_create; /* ERROR */

    std::cout << "Entering main()" << std::endl;
    /*
    {
        auto const& t = Test::instance();
        t.use();
    }
    {
        auto const& t = Test::instance();
        t.use();
    }
    */


  for (int i=0; i<3; i++) {
    create_level();
  }

  leaf_kernel();
  
  std::cout << "Leaving main()" << std::endl;
}

