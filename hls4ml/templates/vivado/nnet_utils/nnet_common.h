#ifndef NNET_COMMON_H_
#define NNET_COMMON_H_

#include "ap_fixed.h"

// This is a substitute for "ceil(n/(float)d)".
#define DIV_ROUNDUP(n, d) ((n + d - 1) / d)
#define MIN(n, d) (n > d ? d : n)
#define MAX(n, d) (n > d ? n : d)

#define STRINGIFY(x) #x
#define EXPAND_STRING(x) STRINGIFY(x)

#ifndef __VITIS_HLS__
#define DATA_PACK_TXT HLS DATA_PACK variable =
#define DATA_PACK_PRAGMA(variable) DATA_PACK_TXT variable
#define PRAGMA_DATA_PACK(variable) _Pragma(EXPAND_STRING(DATA_PACK_PRAGMA(variable)))
#else
#define PRAGMA_DATA_PACK(variable)
#endif

namespace nnet {

// Common type definitions
enum io_type { io_parallel = 0, io_stream };
enum strategy { latency, resource };

/* ---
 * Balanced tree reduce implementation.
 * For use in scenarios where Vivado cannot expression balance
 * Reduces an array of inputs to a single value using the template binary operator 'Op',
 * for example summing all elements with Op_add, or finding the maximum with Op_max
 * Use only when the input array is fully unrolled. Or, slice out a fully unrolled section
 * before applying and accumulate the result over the rolled dimension.
 * --- */
template <class T, int N, class Op> T reduce(const T *x, Op op) {
    static constexpr int leftN = pow2(floorlog2(N - 1)) > 0 ? pow2(floorlog2(N - 1)) : 0;
    static constexpr int rightN = N - leftN > 0 ? N - leftN : 0;
    if (N == 1) {
        return x[0];
    }
    if (N == 2) {
        return op(x[0], x[1]);
    }
    return op(reduce<T, leftN, Op>(x, op), reduce<T, rightN, Op>(x + leftN, op));
}

template <class T> class Op_add {
  public:
    T operator()(T a, T b) { return a + b; }
};

template <class T> class Op_and {
  public:
    T operator()(T a, T b) { return a && b; }
};

template <class T> class Op_or {
  public:
    T operator()(T a, T b) { return a || b; }
};

template <class T> class Op_max {
  public:
    T operator()(T a, T b) { return a >= b ? a : b; }
};

template <class T> class Op_min {
  public:
    T operator()(T a, T b) { return a <= b ? a : b; }
};

} // namespace nnet

#endif
