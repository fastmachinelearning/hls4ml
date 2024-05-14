#ifndef NNET_INSTR_GEN_H_
#define NNET_INSTR_GEN_H_

#include "ap_fixed.h"
#include "nnet_helpers.h"
#include <iostream>

namespace nnet {

template <class data_T, typename CONFIG_T> class FillConv1DBuffer {
  public:
    static void fill_buffer(data_T data[CONFIG_T::in_width * CONFIG_T::n_chan],
                            data_T buffer[CONFIG_T::n_pixels][CONFIG_T::filt_width * CONFIG_T::n_chan],
                            const unsigned partition) {
        // To be implemented in subclasses
    }
};

template <class data_T, typename CONFIG_T> class FillConv2DBuffer {
  public:
    static void
    fill_buffer(data_T data[CONFIG_T::in_height * CONFIG_T::in_width * CONFIG_T::n_chan],
                data_T buffer[CONFIG_T::n_pixels][CONFIG_T::filt_height * CONFIG_T::filt_width * CONFIG_T::n_chan],
                const unsigned partition) {
        // To be implemented in subclasses
    }
};

template <int s, int b, int i, ap_q_mode Q, ap_o_mode O, int N> ap_fixed<b, i + s> bit_shift(ap_fixed<b, i, Q, O, N> x) {
    #pragma HLS INLINE
    ap_fixed<b, i + s> r;
    r.range() = x.range();
    return r;
};

template <int s, int b, int i, ap_q_mode Q, ap_o_mode O, int N> ap_ufixed<b, i + s> bit_shift(ap_ufixed<b, i, Q, O, N> x) {
    #pragma HLS INLINE
    ap_ufixed<b, i + s> r;
    r.range() = x.range();
    return r;
};

template <int s, int b> ap_fixed<b, s> bit_shift(ap_int<b> x) {
    #pragma HLS INLINE
    ap_fixed<b, s> r;
    r.range() = x.range();
    return r;
};

template <int s, int b> ap_ufixed<b, s> bit_shift(ap_uint<b> x) {
    #pragma HLS INLINE
    ap_ufixed<b, s> r;
    r.range() = x.range();
    return r;
};
// hls4ml insert code

} // namespace nnet

#endif
