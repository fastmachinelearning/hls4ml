#include "ap_int.h"
//#include <mc_scverify.h>
#if 0
#ifndef SC_INCLUDE_DYNAMIC_PROCESSES
#define SC_INCLUDE_DYNAMIC_PROCESSES
#endif
#endif


int main() {
  ap_int<10> a = 0;
  a[9] = 1;
  ap_uint<10> b = a << 1;

  std::cout << a << std::endl;
  std::cout << b << std::endl;
  b = a >> 1;
  std::cout << b << std::endl;

  a[3] = 1;
  a[1] = 1;

  b.range(5,2) = a.range(5,1);
  std::cout << std::hex << b << std::dec << std::endl;

  ap_fixed<10,1,AP_RND,AP_SAT> c = 0.5;
  std::cout << c << std::endl;
}
