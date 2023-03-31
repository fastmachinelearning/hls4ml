#ifndef NNET_SEPARABLE_CONV2D_STREAM_H_
#define NNET_SEPARABLE_CONV2D_STREAM_H_

#include "hls_stream.h"
#include "nnet_common.h"
#include "nnet_conv2d_stream.h"
#include "nnet_sepconv_stream.h"

namespace nnet {

// Line Buffer Implementation (Phil's)
template <class data_T, class res_T, typename CONFIG_T>
void depthwise_conv_2d_buffer_cl(
    hls::stream<data_T> &data, hls::stream<res_T> &res,
    typename CONFIG_T::weight_t weights[CONFIG_T::filt_height * CONFIG_T::filt_width * CONFIG_T::n_chan],
    typename CONFIG_T::bias_t biases[CONFIG_T::n_chan]) {
    assert(CONFIG_T::pad_top == 0 && CONFIG_T::pad_bottom == 0 && CONFIG_T::pad_left == 0 && CONFIG_T::pad_right == 0);

    static ap_shift_reg<typename data_T::value_type, CONFIG_T::in_width> line_buffer[CONFIG_T::filt_height - 1]
                                                                                    [CONFIG_T::n_chan];
    #pragma HLS ARRAY_PARTITION variable = line_buffer complete dim = 2

    if (CONFIG_T::strategy == nnet::latency) {
    ReadInputHeight:
        for (unsigned i_ih = 0; i_ih < CONFIG_T::in_height; i_ih++) {
        ReadInputWidth:
            for (unsigned i_iw = 0; i_iw < CONFIG_T::in_width; i_iw++) {
                #pragma HLS LOOP_FLATTEN
                #pragma HLS PIPELINE II=CONFIG_T::reuse_factor
                if (CONFIG_T::filt_height > 1) {
                    compute_depthwise_output_buffer_2d<data_T, res_T, CONFIG_T>(data.read(), line_buffer, res, weights,
                                                                                biases);
                } else {
                    compute_depthwise_output_buffer_1d<data_T, res_T, CONFIG_T>(data.read(), res, weights, biases);
                }
            }
        }
    } else {
    ReadInputHeightSerial:
        for (unsigned i_ih = 0; i_ih < CONFIG_T::in_height; i_ih++) {
        ReadInputWidthSerial:
            for (unsigned i_iw = 0; i_iw < CONFIG_T::in_width; i_iw++) {
                #pragma HLS LOOP_FLATTEN
                if (CONFIG_T::filt_height > 1) {
                    compute_depthwise_output_buffer_2d<data_T, res_T, CONFIG_T>(data.read(), line_buffer, res, weights,
                                                                                biases);
                } else {
                    compute_depthwise_output_buffer_1d<data_T, res_T, CONFIG_T>(data.read(), res, weights, biases);
                }
            }
        }
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void pointwise_conv_2d_cl(hls::stream<data_T> &data, hls::stream<res_T> &res,
                          typename CONFIG_T::weight_t weights[CONFIG_T::n_chan * CONFIG_T::n_filt],
                          typename CONFIG_T::bias_t biases[CONFIG_T::n_filt]) {
    assert(CONFIG_T::pad_top == 0 && CONFIG_T::pad_bottom == 0 && CONFIG_T::pad_left == 0 && CONFIG_T::pad_right == 0);
    assert(CONFIG_T::filt_height == 1 && CONFIG_T::filt_width == 1);

    #pragma HLS ARRAY_PARTITION variable=weights complete
    #pragma HLS ARRAY_PARTITION variable=biases complete

    if (CONFIG_T::strategy == nnet::latency) {
    ReadInputHeight:
        for (unsigned i_ih = 0; i_ih < CONFIG_T::in_height; i_ih++) {
        ReadInputWidth:
            for (unsigned i_iw = 0; i_iw < CONFIG_T::in_width; i_iw++) {
                #pragma HLS LOOP_FLATTEN
                #pragma HLS PIPELINE II=CONFIG_T::reuse_factor
                if (i_ih % CONFIG_T::stride_height == 0 && i_iw % CONFIG_T::stride_width == 0) {
                    pointwise_mult_buffer<data_T, res_T, CONFIG_T>(data.read(), res, weights, biases);
                } else {
                    data.read();
                }
            }
        }
    } else {
    ReadInputHeightSerial:
        for (unsigned i_ih = 0; i_ih < CONFIG_T::in_height; i_ih++) {
        ReadInputWidthSerial:
            for (unsigned i_iw = 0; i_iw < CONFIG_T::in_width; i_iw++) {
                #pragma HLS LOOP_FLATTEN
                if (i_ih % CONFIG_T::stride_height == 0 && i_iw % CONFIG_T::stride_width == 0) {
                    pointwise_mult_buffer<data_T, res_T, CONFIG_T>(data.read(), res, weights, biases);
                } else {
                    data.read();
                }
            }
        }
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void separable_conv_2d_cl(hls::stream<data_T> &data, hls::stream<res_T> &res,
                          typename CONFIG_T::depthwise_config::weight_t
                              depthwise_weights[CONFIG_T::depthwise_config::filt_height *
                                                CONFIG_T::depthwise_config::filt_width * CONFIG_T::depthwise_config::n_chan],
                          typename CONFIG_T::pointwise_config::weight_t
                              pointwise_weights[CONFIG_T::pointwise_config::n_chan * CONFIG_T::pointwise_config::n_filt],
                          typename CONFIG_T::depthwise_config::bias_t depthwise_biases[CONFIG_T::depthwise_config::n_chan],
                          typename CONFIG_T::pointwise_config::bias_t pointwise_biases[CONFIG_T::pointwise_config::n_filt]) {
    assert(CONFIG_T::depthwise_config::implementation == conv_implementation::linebuffer &&
           "Only \"linebuffer\" implementation is supported in Vitis HLS.");
    assert(CONFIG_T::pointwise_config::implementation == conv_implementation::linebuffer &&
           "Only \"linebuffer\" implementation is supported in Vitis HLS.");

    #pragma HLS DATAFLOW

    hls::stream<data_T> depthwise_res;
    unsigned res_depth = CONFIG_T::depthwise_config::out_height * CONFIG_T::depthwise_config::out_width;
    #pragma HLS STREAM variable=depthwise_res depth=res_depth

    depthwise_conv_2d_buffer_cl<data_T, data_T, typename CONFIG_T::depthwise_config>(data, depthwise_res, depthwise_weights,
                                                                                     depthwise_biases);
    pointwise_conv_2d_cl<data_T, res_T, typename CONFIG_T::pointwise_config>(depthwise_res, res, pointwise_weights,
                                                                             pointwise_biases);
}

} // namespace nnet
#endif
