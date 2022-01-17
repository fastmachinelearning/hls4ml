#ifndef NNET_MULT_H_
#define NNET_MULT_H_

#include "nnet_common.h"
#include "nnet_helpers.h"
#include "hls_stream.h"
#include <math.h>
#include <iostream>

namespace nnet {

namespace product{

/* ---
 * different methods to perform the product of input and weight, depending on the
 * types of each.
 * --- */

class Product{
    public:
    static void limit(unsigned multiplier_limit) {} // Nothing to do here
};

template<class x_T, class w_T>
class both_binary : public Product{
    public:
    static x_T product(x_T a, w_T w){
        // specialisation for 1-bit weights and incoming data
        #pragma HLS INLINE
        return a == w;
    }
};

template<class x_T, class w_T>
class weight_binary : public Product{
    public:
    static x_T product(x_T a, w_T w){
        // Specialisation for 1-bit weights, arbitrary data
        #pragma HLS INLINE
        return w == 0 ? (x_T) -a : a;
    }
};

template<class x_T, class w_T>
class data_binary : public Product{
    public:
    static w_T product(x_T a, w_T w){
        // Specialisation for 1-bit data, arbitrary weight
        #pragma HLS INLINE
        return a == 0 ? (w_T) -w : w;
    }
};

template<class x_T, class w_T>
class weight_ternary : public Product{
    public:
    static x_T product(x_T a, w_T w){
        // Specialisation for 2-bit weights, arbitrary data
        #pragma HLS INLINE
        if (w == 0) return (x_T) 0;
        else if(w == -1) return (x_T) -a;
        else return (x_T) a; // if(w == 1)
    }
};

template<class x_T, class w_T>
class mult : public Product{
    public:
    static auto product(x_T a, w_T w) -> decltype(a*w)
    {
        // 'Normal' product
        #pragma HLS INLINE
        return a * w;
    }
    static void limit(unsigned multiplier_limit){
        #pragma HLS INLINE
        #pragma HLS ALLOCATION instances=mul limit=multiplier_limit operation
    }
};

template<class x_T, class w_T>
class weight_exponential : public Product{
    public:
    using rt = x_T;
    static rt product(x_T a, w_T w){
        std::cerr << "Should not match to this function" << std::endl;
        // Shift product for exponential weights
        #pragma HLS INLINE
        // shift by the exponent. Negative weights shift right
        rt y = static_cast<rt>(a) << w.weight;
        // negate or not depending on weight sign
        return w.sign == 1 ? y : static_cast<rt>(-y);
    }
};

template<class w_T, int _AP_W>
class weight_exponential<ap_int<_AP_W>, w_T> : public Product{
    public:
    using rt = ap_fixed<_AP_W + 2*decltype(w_T::weight)::width, _AP_W + decltype(w_T::weight)::width>;
    static rt product(ap_int<_AP_W> a, w_T w){
        // Shift product for exponential weights
        #pragma HLS INLINE
        // shift by the exponent. Negative weights shift right
        rt y = static_cast<rt>(a) << w.weight;
        // negate or not depending on weight sign
        return w.sign == 1 ? y : static_cast<rt>(-y);
    }
};

template<class w_T, int _AP_W>
class weight_exponential<ap_uint<_AP_W>, w_T> : public Product{
    public:
    using rt = ap_fixed<_AP_W + 2*decltype(w_T::weight)::width + 1, _AP_W + decltype(w_T::weight)::width + 1>;
    static rt product(ap_uint<_AP_W> a, w_T w){
        // Shift product for exponential weights
        #pragma HLS INLINE
        // shift by the exponent. Negative weights shift right
        rt y = static_cast<rt>(a) << w.weight;
        // negate or not depending on weight sign
        return w.sign == 1 ? y : static_cast<rt>(-y);
    }
};

template<class w_T, int _AP_W, int _AP_I, ap_q_mode _AP_Q, ap_o_mode _AP_O, int _AP_N>
class weight_exponential<ap_fixed<_AP_W,_AP_I,_AP_Q, _AP_O, _AP_N>, w_T> : public Product{
    public:
    using rt = ap_fixed<_AP_W + 2*decltype(w_T::weight)::width, _AP_I + decltype(w_T::weight)::width,
                        _AP_Q, _AP_O, _AP_N>;
    static rt product(ap_fixed<_AP_W,_AP_I,_AP_Q, _AP_O, _AP_N> a, w_T w){
        // Shift product for exponential weights
        #pragma HLS INLINE
        // shift by the exponent. Negative weights shift right
        rt y = static_cast<rt>(a) << w.weight;
        // negate or not depending on weight sign
        return w.sign == 1 ? y : static_cast<rt>(-y);
    }
};

template<class w_T, int _AP_W, int _AP_I, ap_q_mode _AP_Q, ap_o_mode _AP_O, int _AP_N>
class weight_exponential<ap_ufixed<_AP_W,_AP_I,_AP_Q, _AP_O, _AP_N>, w_T> : public Product{
    public:
    using rt = ap_fixed<_AP_W + 2*decltype(w_T::weight)::width + 1, _AP_I + decltype(w_T::weight)::width + 1,
                        _AP_Q, _AP_O, _AP_N>;
    static rt product(ap_ufixed<_AP_W,_AP_I,_AP_Q, _AP_O, _AP_N> a, w_T w){
        // Shift product for exponential weights
        #pragma HLS INLINE
        // shift by the exponent. Negative weights shift right
        // shift by the exponent. Negative weights shift right
        rt y = static_cast<rt>(a) << w.weight;
        // negate or not depending on weight sign
        return w.sign == 1 ? y : static_cast<rt>(-y);
    }
};

} // namespace product_type

template<class data_T, class res_T, typename CONFIG_T>
inline typename std::enable_if<std::is_same<data_T, ap_uint<1>>::value
        && std::is_same<typename CONFIG_T::weight_t, ap_uint<1>>::value, ap_int<nnet::ceillog2(CONFIG_T::n_in) + 2>>::type
cast(typename CONFIG_T::accum_t x){
  return (ap_int<nnet::ceillog2(CONFIG_T::n_in) + 2>) (x - CONFIG_T::n_in / 2) * 2;
}

template<class data_T, class res_T, typename CONFIG_T>
inline typename std::enable_if<std::is_same<data_T, ap_uint<1>>::value
        && ! std::is_same<typename CONFIG_T::weight_t, ap_uint<1>>::value, res_T>::type
cast(typename CONFIG_T::accum_t x){
  return (res_T) x;
}

template<class data_T, class res_T, typename CONFIG_T>
inline typename std::enable_if<(! std::is_same<data_T, ap_uint<1>>::value), res_T>::type
cast(typename CONFIG_T::accum_t x){
  return (res_T) x;
}

}

#endif
