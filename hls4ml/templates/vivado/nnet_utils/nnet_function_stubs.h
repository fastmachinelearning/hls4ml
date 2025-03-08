#ifndef NNET_FUNCTION_STUBS_H_
#define NNET_FUNCTION_STUBS_H_

#include "nnet_helpers.h"

#include "hls_stream.h"
#include "nnet_common.h"
#include "nnet_mult.h"

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

template <class data_T, class res_T, typename CONFIG_T> class DenseKernel {
  public:
    static void dense(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_out],
                      typename CONFIG_T::weight_t weights[CONFIG_T::n_in * CONFIG_T::n_out],
                      typename CONFIG_T::bias_t biases[CONFIG_T::n_out]) {
        // To be implemented in subclasses
    }
};

template <class data_T, class res_T, typename CONFIG_T> class DepthwiseDenseKernel {
  public:
    static void dense(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_out],
                      typename CONFIG_T::weight_t weights[CONFIG_T::n_in * CONFIG_T::n_out],
                      typename CONFIG_T::bias_t biases[CONFIG_T::n_out]) {
        // To be implemented in subclasses
    }
};

template <class data_T, class res_T, typename CONFIG_T> class Conv1DKernel {
  public:
    static void conv(data_T data[CONFIG_T::in_width * CONFIG_T::n_chan], res_T res[CONFIG_T::out_width * CONFIG_T::n_filt],
                     typename CONFIG_T::weight_t weights[CONFIG_T::n_chan * CONFIG_T::n_filt],
                     typename CONFIG_T::bias_t biases[CONFIG_T::n_filt]) {
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
