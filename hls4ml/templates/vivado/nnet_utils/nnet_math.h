#ifndef NNET_MATH_H_
#define NNET_MATH_H_

#include "hls_math.h"

namespace nnet {

// This header defines the functions that return type different from the input
// For example, hls::sin(x) returns ap_fixed<W-I+2,2>
// By ensuring we return the same type we can avoid casting issues in expressions

template<typename T>
T sinpi(T x) {
    return (T) hls::sinpi(x);
};

template<typename T>
T cospi(T x) {
    return (T) hls::cospi(x);
};

template<typename T>
T sin(T x) {
    return (T) hls::sin(x);
};

template<typename T>
T cos(T x) {
    return (T) hls::cos(x);
};

template<typename T>
T asin(T x) {
    return (T) hls::asin(x);
};

template<typename T>
T acos(T x) {
    return (T) hls::acos(x);
};

template<typename T>
T atan(T x) {
    return (T) hls::atan(x);
};

template<typename T>
T atan2(T x) {
    return (T) hls::atan2(x);
};

}

#endif