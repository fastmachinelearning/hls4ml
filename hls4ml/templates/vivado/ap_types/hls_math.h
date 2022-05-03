#ifndef X_HLS_MATH_H
#define X_HLS_MATH_H

#include <cmath>
#include "ap_fixed.h"

namespace hls {

template<class T>
static T exp(const T x) {
  return (T) std::exp(x.to_double());
}

}
#endif
