#ifndef X_HLS_MATH_H
#define X_HLS_MATH_H

#include <cmath>
#include "ap_fixed.h"

namespace hls {

template<class T>
static T exp(const T x) {
  return (T) std::exp(x.to_double());
}

template <typename T> T sin(T x) { return (T) std::sin(x.to_double()); };

template <typename T> T cos(T x) { return (T) std::cos(x.to_double()); };

template <typename T> T asin(T x) { return (T) std::asin(x.to_double()); };

template <typename T> T acos(T x) { return (T) std::acos(x.to_double()); };

template <typename T> T atan(T x) { return (T) std::atan(x.to_double()); };

template <typename T> T atan2(T x, T y) { return (T) hls::atan2(x.to_double(), y.to_double()); };

}
#endif
