#ifndef NNET_SEPARABLE_CONV1D_STREAM_H_
#define NNET_SEPARABLE_CONV1D_STREAM_H_

#include "nnet_common.h"
#include "hls_stream.h"
#include "nnet_sepconv_stream.h"
#include "nnet_conv1d_stream.h"

namespace nnet {

template<class data_T, class res_T, typename CONFIG_T>
void depthwise_conv_1d_encoded_cl(
    hls::stream<data_T> &data,
    hls::stream<res_T>  &res,
    typename CONFIG_T::weight_t weights[CONFIG_T::filt_width * CONFIG_T::n_chan],
    typename CONFIG_T::bias_t   biases[CONFIG_T::n_chan])
{
    assert(CONFIG_T::pad_left == 0 && CONFIG_T::pad_right == 0);

    hls::stream<typename data_T::value_type> data_window[CONFIG_T::filt_width * CONFIG_T::n_chan];
    const int win_depth = CONFIG_T::out_width;
    for (unsigned i_out = 0; i_out < CONFIG_T::filt_width * CONFIG_T::n_chan; i_out++) {
        #pragma HLS STREAM variable=data_window[i_out] depth=win_depth
    }

    #pragma HLS ARRAY_PARTITION variable=CONFIG_T::pixels complete

    res_T res_pack;
    #pragma HLS DATA_PACK variable=res_pack
    unsigned outputs_ready = 0;

    ap_uint<CONFIG_T::filt_width> pixel_idx[data_T::size / CONFIG_T::n_chan];
    #pragma HLS ARRAY_PARTITION variable=pixel_idx complete

    ReadInputWidth: for (unsigned i_iw = 0; i_iw < CONFIG_T::in_width / (data_T::size / CONFIG_T::n_chan); i_iw++) {
        #pragma HLS LOOP_FLATTEN
        if (CONFIG_T::strategy == nnet::latency && data_T::size / CONFIG_T::n_chan == 1) {
            #pragma HLS PIPELINE II=CONFIG_T::reuse_factor
        }
        compute_scaled_indices_1d<data_T, CONFIG_T>(i_iw, pixel_idx);
        compute_depthwise_output_encoded<data_T, res_T, CONFIG_T>(data.read(), data_window, res, res_pack, outputs_ready, weights, biases, pixel_idx);
    }
}

template<class data_T, class res_T, typename CONFIG_T>
void depthwise_conv_1d_buffer_cl(
    hls::stream<data_T> &data,
    hls::stream<res_T>  &res,
    typename CONFIG_T::weight_t weights[CONFIG_T::filt_width * CONFIG_T::n_chan],
    typename CONFIG_T::bias_t   biases[CONFIG_T::n_chan])
{
    assert(CONFIG_T::pad_left == 0 && CONFIG_T::pad_right == 0);

    ReadInputWidth: for (unsigned i_iw = 0; i_iw < CONFIG_T::in_width; i_iw++) {
        #pragma HLS LOOP_FLATTEN
        if (CONFIG_T::strategy == nnet::latency) {
            #pragma HLS PIPELINE II=CONFIG_T::reuse_factor
        }
        compute_depthwise_output_buffer_1d<data_T, res_T, CONFIG_T>(data.read(), res, weights, biases);
    }
}

template<class data_T, class res_T, typename CONFIG_T>
void pointwise_conv_1d_cl(
    hls::stream<data_T> &data,
    hls::stream<res_T>  &res,
    typename CONFIG_T::weight_t weights[CONFIG_T::n_chan * CONFIG_T::n_filt],
    typename CONFIG_T::bias_t   biases[CONFIG_T::n_filt])
{
    assert(CONFIG_T::pad_left == 0 && CONFIG_T::pad_right == 0);
    assert(CONFIG_T::filt_width == 1);

    #pragma HLS ARRAY_PARTITION variable=weights complete
    #pragma HLS ARRAY_PARTITION variable=biases complete

    ReadInputWidth: for (unsigned i_iw = 0; i_iw < CONFIG_T::in_width / (data_T::size / CONFIG_T::n_chan); i_iw++) {
        if (CONFIG_T::strategy == nnet::latency && data_T::size / CONFIG_T::n_chan == 1) {
            #pragma HLS PIPELINE II=CONFIG_T::reuse_factor
        }
        if (i_iw % CONFIG_T::stride_width == 0) {
            pointwise_mult_buffer<data_T, res_T, CONFIG_T>(data.read(), res, weights, biases);
        } else {
            data.read();
        }
    }
}


template<class data_T, class res_T, typename CONFIG_T>
void separable_conv_1d_cl(
    hls::stream<data_T> &data,
    hls::stream<res_T>  &res,
    typename CONFIG_T::depthwise_config::weight_t depthwise_weights[CONFIG_T::depthwise_config::filt_width * CONFIG_T::depthwise_config::n_chan],
    typename CONFIG_T::pointwise_config::weight_t pointwise_weights[CONFIG_T::pointwise_config::n_chan * CONFIG_T::pointwise_config::n_filt],
    typename CONFIG_T::depthwise_config::bias_t   depthwise_biases[CONFIG_T::depthwise_config::n_chan],
    typename CONFIG_T::pointwise_config::bias_t   pointwise_biases[CONFIG_T::pointwise_config::n_filt]
) {
    #pragma HLS DATAFLOW

    hls::stream<data_T> depthwise_res;
    unsigned res_depth = CONFIG_T::depthwise_config::out_width;
    #pragma HLS STREAM variable=depthwise_res depth=res_depth

    switch(CONFIG_T::depthwise_config::implementation){
        case conv_implementation::linebuffer:
            depthwise_conv_1d_buffer_cl<data_T, data_T, typename CONFIG_T::depthwise_config>(data, depthwise_res, depthwise_weights, depthwise_biases);
            break;
        case conv_implementation::encoded:
            depthwise_conv_1d_encoded_cl<data_T, data_T, typename CONFIG_T::depthwise_config>(data, depthwise_res, depthwise_weights, depthwise_biases);
            break;
    } 
    pointwise_conv_1d_cl<data_T, res_T, typename CONFIG_T::pointwise_config>(depthwise_res, res, pointwise_weights, pointwise_biases);
}

}
#endif
