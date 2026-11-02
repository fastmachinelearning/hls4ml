#ifndef NNET_MULT_H_
#define NNET_MULT_H_

#include "hls_stream.h"
#include "nnet_common.h"
#include "nnet_helpers.h"
#include <iostream>
#include <math.h>

namespace nnet {

namespace product {

/* ---
 * different methods to perform the product of input and weight, depending on the
 * types of each.
 * --- */

class Product {
  public:
    // The value a sum of n products starts from
    template <class acc_T, class b_T> static acc_T init(b_T b, int n) { return static_cast<acc_T>(b); }
};

template <class x_T, class w_T> class both_binary : public Product {
  public:
    // With products of 2 for equal data and weight (both +1 or both -1) and 0 otherwise, a sum of n products started at
    // bias - n is the bias plus the sum of the products of the -1 and +1 values
    template <class acc_T, class b_T> static acc_T init(b_T b, int n) {
        return static_cast<acc_T>(b) - static_cast<acc_T>(n);
    }
    static ap_uint<2> product(x_T a, w_T w) {
        // specialisation for 1-bit weights and incoming data
        #pragma HLS INLINE
        return a == w ? 2 : 0;
    }
};

// The products of the -1 and +1 values, for sums that do not start at bias - n
template <class x_T, class w_T> class both_binary_signed : public Product {
  public:
    static ap_int<2> product(x_T a, w_T w) {
        // specialisation for 1-bit weights and incoming data
        #pragma HLS INLINE
        return a == w ? 1 : -1;
    }
};

template <class x_T, class w_T> class weight_binary : public Product {
  public:
    static auto product(x_T a, w_T w) -> decltype(-a) {
        // Specialisation for 1-bit weights, arbitrary data
        #pragma HLS INLINE
        if (w == 0)
            return -a;
        else
            return a;
    }
};

template <class x_T, class w_T> class data_binary : public Product {
  public:
    static auto product(x_T a, w_T w) -> decltype(-w) {
        // Specialisation for 1-bit data, arbitrary weight
        #pragma HLS INLINE
        if (a == 0)
            return -w;
        else
            return w;
    }
};

template <class x_T, class w_T> class weight_ternary : public Product {
  public:
    static auto product(x_T a, w_T w) -> decltype(-a) {
        // Specialisation for 2-bit weights, arbitrary data
        #pragma HLS INLINE
        if (w == 0)
            return 0;
        else if (w == -1)
            return -a;
        else
            return a; // if(w == 1)
    }
};

template <class x_T, class w_T> class mult : public Product {
  public:
    static auto product(x_T a, w_T w) -> decltype(a * w) {
        // 'Normal' product
        #pragma HLS INLINE
        return a * w;
    }
};

template <class x_T, class w_T> class weight_exponential : public Product {
  public:
    using r_T = ap_fixed<2 * (decltype(w_T::weight)::width + x_T::width), (decltype(w_T::weight)::width + x_T::width)>;
    static r_T product(x_T a, w_T w) {
        // Shift product for exponential weights
        #pragma HLS INLINE

        // Shift by the exponent. Negative weights shift right
        r_T y = static_cast<r_T>(a) << w.weight;

        // Negate or not depending on weight sign
        return w.sign == 1 ? y : static_cast<r_T>(-y);
    }
};

template <class x_T, class w_T> class data_binary_weight_exponential : public Product {
  public:
    // Holds 2^e exactly for every exponent e of the weight type
    static const int max_shift = 1 << (decltype(w_T::weight)::width - 1);
    using r_T = ap_fixed<2 * max_shift + 1, max_shift + 1>;
    static r_T product(x_T a, w_T w) {
        // Specialisation for 1-bit data (0 stands for -1) and exponential weights
        #pragma HLS INLINE
        r_T y = static_cast<r_T>(1) << w.weight;
        return (a == 1) == (w.sign == 1) ? y : static_cast<r_T>(-y);
    }
};

} // namespace product

template <class data_T, typename CONFIG_T> inline typename CONFIG_T::accum_t init(typename CONFIG_T::bias_t b) {
    return CONFIG_T::template product<data_T, typename CONFIG_T::weight_t>::template init<typename CONFIG_T::accum_t>(
        b, CONFIG_T::n_in);
}

template <class data_T, class res_T, typename CONFIG_T> inline res_T cast(typename CONFIG_T::accum_t x) { return (res_T)x; }

} // namespace nnet

#endif
