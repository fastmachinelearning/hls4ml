#ifndef NNET_COMMON_H_
#define NNET_COMMON_H_

#include "nnet_helpers.h"
#include <sycl/ext/intel/ac_types/ac_fixed.hpp>
#include <sycl/ext/intel/ac_types/ac_fixed_math.hpp>
#include <sycl/ext/intel/ac_types/ac_int.hpp>

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
    static constexpr int leftN = pow2<floorlog2<N - 1>::val>::val > 0 ? pow2<floorlog2<N - 1>::val>::val : 0;
    static constexpr int rightN = N - leftN > 0 ? N - leftN : 0;
    if constexpr (N == 1) {
        return x[0];
    } else if constexpr (N == 2) {
        return op(x[0], x[1]);
    } else {
        return op(reduce<T, leftN, Op>(x, op), reduce<T, rightN, Op>(x + leftN, op));
    }
}

// alternate reduce - basic
// template <class T, int N, class Op> T reduce(const T *x, Op op) {
//     if (N == 1) {
//         return x[0];
//     }
//     auto val = op(x[0], x[1]);
//     for (int i = 2; i < N; i++) {
//         val = op(val, x[i]);
//     }
//     return val;
// }

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
