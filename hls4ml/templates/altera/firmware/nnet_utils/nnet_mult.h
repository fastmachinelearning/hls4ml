#ifndef NNET_MULT_H_
#define NNET_MULT_H_

#include "nnet_common.h"
#include "nnet_helpers.h"
#include <math.h>

namespace nnet {

//  Different methods to perform the product of input and weight, depending on their types.
namespace product {

class Product {
  public:
    static void limit(unsigned multiplier_limit) {}
    template <class acc_T, class b_T> static acc_T init(b_T b, int n) { return static_cast<acc_T>(b); }
};

template <class x_T, class w_T> class both_binary : public Product {
  public:
    // With products of 2 for equal data and weight (both +1 or both -1) and 0 otherwise, a sum of n products started at
    // bias - n is the bias plus the sum of the products of the -1 and +1 values
    template <class acc_T, class b_T> static acc_T init(b_T b, int n) {
        return static_cast<acc_T>(b) - static_cast<acc_T>(n);
    }
    inline static ac_int<2, false> product(x_T a, w_T w) {
        // specialisation for 1-bit weights and incoming data
        return a == w ? 2 : 0;
    }
};

template <class x_T, class w_T> class weight_binary : public Product {
  public:
    inline static auto product(x_T a, w_T w) -> decltype(-a) {
        // Specialisation for 1-bit weights, arbitrary data
        if (w == 0)
            return -a;
        else
            return a;
    }
};

template <class x_T, class w_T> class data_binary : public Product {
  public:
    inline static auto product(x_T a, w_T w) -> decltype(-w) {
        // Specialisation for 1-bit data, arbitrary weight
        if (a == 0)
            return -w;
        else
            return w;
    }
};

template <class x_T, class w_T> class weight_ternary : public Product {
  public:
    inline static auto product(x_T a, w_T w) -> decltype(-a) {
        // Specialisation for 2-bit weights, arbitrary data
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
    inline static auto product(x_T a, w_T w) -> decltype(a * w) {
        // 'Normal' product
        return a * w;
    }
    static void limit(unsigned multiplier_limit) {
        // TODO: Implement for Quartus
        // #pragma HLS ALLOCATION instances=mul limit=multiplier_limit operation > Vivado-only, replace with Intel HLS
        // pragma
    }
};

template <class x_T, class w_T> class weight_exponential : public Product {
  public:
    using r_T = ac_fixed<2 * (w_T::second_type::width + x_T::width), (w_T::second_type::width + x_T::width), true>;
    inline static r_T product(x_T a, w_T w) {
        // Shift product for exponential weights
        // Shift by the exponent. Negative weights shift right
        r_T y = static_cast<r_T>(a) << w.second;

        // Negate or not depending on weight sign
        return w.first == 1 ? y : static_cast<r_T>(-y);
    }
};
template <class x_T, class w_T> class data_binary_weight_exponential : public Product {
  public:
    // Holds 2^e exactly for every exponent e of the weight type
    static const int max_shift = 1 << (w_T::second_type::width - 1);
    using r_T = ac_fixed<2 * max_shift + 1, max_shift + 1, true>;
    inline static r_T product(x_T a, w_T w) {
        // Specialisation for 1-bit data (0 stands for -1) and exponential weights
        r_T y = static_cast<r_T>(1) << w.second;
        return (a == 1) == (w.first == 1) ? y : static_cast<r_T>(-y);
    }
};
} // namespace product

template <class data_T, typename CONFIG_T> inline typename CONFIG_T::accum_t init(typename CONFIG_T::bias_t::value_type b) {
    return CONFIG_T::template product<data_T, typename CONFIG_T::weight_t::value_type>::template init<
        typename CONFIG_T::accum_t>(b, CONFIG_T::n_in);
}

template <class data_T, class res_T, typename CONFIG_T> inline res_T cast(typename CONFIG_T::accum_t x) {
    return static_cast<res_T>(x);
}

} // namespace nnet

#endif
