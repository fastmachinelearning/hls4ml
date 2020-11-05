#ifndef NNET_CONV2D_STREAM_H_
#define NNET_CONV2D_STREAM_H_

#include "nnet_common.h"
#include "hls_stream.h"

namespace nnet {

template<unsigned K, unsigned S, unsigned W>
unsigned scale_index(const unsigned idx) {
    #pragma HLS INLINE

    if (idx < K - S) {
        return idx;
    }

    constexpr unsigned nW = ((W - K) / S) * S + K; // Nearest W without unused pixels on the right
    constexpr unsigned sW = (DIV_ROUNDUP(K, S) - 1) * S + K; // Scaled W that behaves like original W
    if (idx >= nW) {
        return sW;
    }

    const unsigned r = nW - idx;
    if (r <= K - S) {
        return sW - r;
    }

    return K - S + (idx - (K - S)) % S;
}

template<class data_T, typename CONFIG_T>
void compute_scaled_indices(
    const unsigned h_idx,
    const unsigned w_idx,
    ap_uint<CONFIG_T::filt_height * CONFIG_T::filt_width> *pixel_idx
) {
    const unsigned sh_idx = scale_index<CONFIG_T::filt_height, CONFIG_T::stride_height, CONFIG_T::in_height>(h_idx);
    unsigned wp_idx = w_idx * (data_T::size / CONFIG_T::n_chan);

    ComputeIndex: for (unsigned p = 0; p < data_T::size / CONFIG_T::n_chan; p++) {
        #pragma HLS UNROLL

        unsigned sw_idx = scale_index<CONFIG_T::filt_width, CONFIG_T::stride_width, CONFIG_T::in_width>(wp_idx + p);
        pixel_idx[p] = CONFIG_T::pixels[sh_idx * CONFIG_T::min_width + sw_idx];
    }
}

template<class data_T, class res_T, typename CONFIG_T>
void mult_buffer(
    hls::stream<typename data_T::value_type> data_window[CONFIG_T::filt_height * CONFIG_T::filt_width * CONFIG_T::n_chan],
    res_T& res_pack,
    hls::stream<res_T>& res_stream,
    unsigned & outputs_ready,
    typename CONFIG_T::weight_t weights[CONFIG_T::filt_height * CONFIG_T::filt_width * CONFIG_T::n_chan * CONFIG_T::n_filt],
    typename CONFIG_T::bias_t biases[CONFIG_T::n_filt]
) {
    #pragma HLS INLINE

    typename data_T::value_type data[CONFIG_T::filt_height * CONFIG_T::filt_width * CONFIG_T::n_chan];
    #pragma HLS ARRAY_PARTITION variable=data complete
    typename res_T::value_type res[CONFIG_T::n_filt];
    #pragma HLS ARRAY_PARTITION variable=res complete

    InitData: for (int id = 0; id < CONFIG_T::filt_height * CONFIG_T::filt_width * CONFIG_T::n_chan; id++) {
        #pragma HLS UNROLL
        data[id] = data_window[id].read();
    }

    #pragma HLS INLINE region
    if (CONFIG_T::strategy == nnet::latency) {
        dense_latency<typename data_T::value_type, typename res_T::value_type, typename CONFIG_T::mult_config>(data, res, weights, biases);
    } else {
        dense_resource<typename data_T::value_type, typename res_T::value_type, typename CONFIG_T::mult_config>(data, res, weights, biases);
    }

    CastLoop: for (unsigned jj = 0; jj < CONFIG_T::n_filt; jj++) {
        #pragma HLS UNROLL
        if (res_T::size / CONFIG_T::n_filt == 1) {
            res_pack[jj] = res[jj];
        } else {
            res_pack[outputs_ready * CONFIG_T::n_filt + jj] = res[jj];
        }
    }

    if (res_T::size / CONFIG_T::n_filt == 1) {
        res_stream.write(res_pack);
    } else {
        if (outputs_ready == (res_T::size / CONFIG_T::n_filt) - 1) {
            res_stream.write(res_pack);
            outputs_ready = 0;
        } else {
            outputs_ready++;
        }
    }
}

template<class data_T, class res_T, typename CONFIG_T>
void compute_output(
    const data_T& in_elem,
    hls::stream<typename data_T::value_type> data_window[CONFIG_T::filt_height * CONFIG_T::filt_width * CONFIG_T::n_chan],
    hls::stream<res_T> &res,
    res_T &res_pack,
    unsigned &outputs_ready,
    typename CONFIG_T::weight_t weights[CONFIG_T::filt_height * CONFIG_T::filt_width * CONFIG_T::n_chan * CONFIG_T::n_filt],
    typename CONFIG_T::bias_t biases[CONFIG_T::n_filt],
    ap_uint<CONFIG_T::filt_height * CONFIG_T::filt_width> *pixel_idx
) {
    #pragma HLS INLINE

    MultLoop: for (unsigned p = 0; p < data_T::size / CONFIG_T::n_chan; p++) {
        #pragma HLS PIPELINE II=CONFIG_T::reuse_factor
        CopyDataFilt: for (unsigned f = 0; f < CONFIG_T::filt_height * CONFIG_T::filt_width; f++) {
            #pragma HLS UNROLL
            CopyDataChan: for (unsigned c = 0; c < CONFIG_T::n_chan; c++) {
                #pragma HLS UNROLL
                if (pixel_idx[p][f]) data_window[f * CONFIG_T::n_chan + c].write(in_elem[p * CONFIG_T::n_chan + c]);
            }
        }
        if (pixel_idx[p][CONFIG_T::filt_height * CONFIG_T::filt_width - 1]) {
            mult_buffer<data_T, res_T, CONFIG_T>(data_window, res_pack, res, outputs_ready, weights, biases);
        }
    }
}

template<class data_T, class res_T, typename CONFIG_T>
void conv_2d_cl(
    hls::stream<data_T> &data,
    hls::stream<res_T>  &res,
    typename CONFIG_T::weight_t weights[CONFIG_T::filt_height * CONFIG_T::filt_width * CONFIG_T::n_chan * CONFIG_T::n_filt],
    typename CONFIG_T::bias_t   biases[CONFIG_T::n_filt])
{
    assert(CONFIG_T::pad_top == 0 && CONFIG_T::pad_bottom == 0 && CONFIG_T::pad_left == 0 && CONFIG_T::pad_right == 0);
    assert(CONFIG_T::filt_height == CONFIG_T::filt_width);
    assert(CONFIG_T::stride_height <= CONFIG_T::filt_height && CONFIG_T::stride_width <= CONFIG_T::filt_width);

    hls::stream<typename data_T::value_type> data_window[CONFIG_T::filt_height * CONFIG_T::filt_width * CONFIG_T::n_chan];
    const int win_depth = CONFIG_T::filt_height * CONFIG_T::out_width;
    for (unsigned i_out = 0; i_out < CONFIG_T::filt_height * CONFIG_T::filt_width * CONFIG_T::n_chan; i_out++) {
        #pragma HLS STREAM variable=data_window[i_out] depth=win_depth
    }

    #pragma HLS ARRAY_PARTITION variable=CONFIG_T::pixels complete

    res_T res_pack;
    #pragma HLS DATA_PACK variable=res_pack
    unsigned outputs_ready = 0;

    ap_uint<CONFIG_T::filt_height * CONFIG_T::filt_width> pixel_idx[data_T::size / CONFIG_T::n_chan];
    #pragma HLS ARRAY_PARTITION variable=pixel_idx complete

    ReadInputHeight: for (unsigned i_ih = 0; i_ih < CONFIG_T::in_height; i_ih++) {
        ReadInputWidth: for (unsigned i_iw = 0; i_iw < CONFIG_T::in_width / (data_T::size / CONFIG_T::n_chan); i_iw++) {
            #pragma HLS LOOP_FLATTEN
            if (CONFIG_T::strategy == nnet::latency && data_T::size / CONFIG_T::n_chan == 1) {
                #pragma HLS PIPELINE II=CONFIG_T::reuse_factor
            }
            compute_scaled_indices<data_T, CONFIG_T>(i_ih, i_iw, pixel_idx);
            compute_output<data_T, res_T, CONFIG_T>(data.read(), data_window, res, res_pack, outputs_ready, weights, biases, pixel_idx);
        }
    }
}

}
#endif
