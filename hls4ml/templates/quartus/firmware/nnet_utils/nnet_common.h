#ifndef NNET_COMMON_H_
#define NNET_COMMON_H_

#ifndef __INTELFPGA_COMPILER__
#include "ac_fixed.h"
#include "ac_int.h"
#include "math.h"
#else
#include "HLS/ac_fixed.h"
#include "HLS/ac_int.h"
#include "HLS/math.h"
#endif

#include "nnet_helpers.h"

typedef ac_fixed<16, 6> table_default_t;

namespace nnet {

// Common type definitions
enum io_type { io_parallel = 0, io_stream };

// Default data types (??) TODO: Deprecate
typedef ac_fixed<16, 4> weight_t_def;
typedef ac_fixed<16, 4> bias_t_def;
typedef ac_fixed<32, 10> accum_t_def;

template <class data_T, int NIN1, int NIN2> void merge(data_T data1[NIN1], data_T data2[NIN2], data_T res[NIN1 + NIN2]) {
    #pragma unroll
    for (int ii = 0; ii < NIN1; ii++) {
        res[ii] = data1[ii];
    }
    #pragma unroll
    for (int ii = 0; ii < NIN2; ii++) {
        res[NIN1 + ii] = data2[ii];
    }
}

/* ---
 * Balanced tree reduce implementation.
 * For use in scenarios where Quartus cannot expression balance
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

template <class T> class Op_max {
  public:
    T operator()(T a, T b) { return a >= b ? a : b; }
};

} // namespace nnet

#endif
