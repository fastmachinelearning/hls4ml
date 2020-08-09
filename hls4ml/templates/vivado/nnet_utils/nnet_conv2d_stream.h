#ifndef NNET_CONV2D_STREAM_H_
#define NNET_CONV2D_STREAM_H_

#include "nnet_common.h"
#include "hls_stream.h"

namespace nnet {

template<unsigned K, unsigned W>
unsigned scale_index(const unsigned idx) {
    #pragma HLS INLINE

    if (idx < K - 1) {
        return idx;
    }

    const unsigned r = W - idx;
    if (r <= K - 1) {
        constexpr unsigned sW = 2 * K - 1;
        return sW - r;
    }

    return K - 1;
}

template<typename CONFIG_T>
void compute_scaled_indices(
    const unsigned h_idx,
    const unsigned w_idx,
    ap_uint<CONFIG_T::filt_height * CONFIG_T::filt_width> pixel_idx[CONFIG_T::in_pack_factor]
) {
    constexpr unsigned sin_width = 2 * CONFIG_T::filt_width - 1;

    const unsigned sh_idx = scale_index<CONFIG_T::filt_height, CONFIG_T::in_height>(h_idx);
    unsigned wp_idx = w_idx * CONFIG_T::in_pack_factor;

    ComputeIndex: for (unsigned p = 0; p < CONFIG_T::in_pack_factor; p++) {
        #pragma HLS UNROLL

        unsigned sw_idx = scale_index<CONFIG_T::filt_width, CONFIG_T::in_width>(wp_idx + p);
        pixel_idx[p] = CONFIG_T::pixels[sh_idx * sin_width + sw_idx];
    }
}

template<class data_T, typename CONFIG_T>
void fill_buffer(
    const data_T& in_elem,
    ap_uint<CONFIG_T::filt_height * CONFIG_T::filt_width> pixel_idx[CONFIG_T::in_pack_factor],
    hls::stream<typename data_T::value_type> data_window[CONFIG_T::filt_height * CONFIG_T::filt_width * CONFIG_T::n_chan]
) {

    CopyDataPack: for (unsigned p = 0; p < CONFIG_T::in_pack_factor; p++) {
        #pragma HLS PIPELINE
        CopyDataFilt: for (unsigned f = 0; f < CONFIG_T::filt_height * CONFIG_T::filt_width; f++) {
            CopyDataChan: for (unsigned c = 0; c < CONFIG_T::n_chan; c++) {
                if (pixel_idx[p][f]) data_window[f * CONFIG_T::n_chan + c].write(in_elem[p * CONFIG_T::n_chan + c]);
            }
        }
    }
}

template<class data_T, class res_T, typename CONFIG_T>
void mult_buffer(
    hls::stream<typename data_T::value_type> data_window[CONFIG_T::filt_height * CONFIG_T::filt_width * CONFIG_T::n_chan],
    hls::stream<res_T> &res,
	res_T &res_pack,
    unsigned &outputs_ready,
    typename CONFIG_T::weight_t weights[CONFIG_T::filt_height * CONFIG_T::filt_width * CONFIG_T::n_chan * CONFIG_T::n_filt],
    typename CONFIG_T::bias_t biases[CONFIG_T::n_filt]
) {
    #pragma HLS INLINE

    constexpr int n_in = CONFIG_T::filt_height * CONFIG_T::filt_width * CONFIG_T::n_chan;
    constexpr int n_out = CONFIG_T::n_filt;

    typename CONFIG_T::accum_t mult[n_in * n_out];
    typename CONFIG_T::accum_t acc[n_out];
    #pragma HLS ARRAY_PARTITION variable=mult complete
    #pragma HLS ARRAY_PARTITION variable=acc complete

    ResetResBuffer: for (unsigned f = 0; f < n_out; f++) {
        #pragma HLS UNROLL
        //out_pixel[f] = biases[f];
        acc[f] = biases[f];
    }

    ProductInLoop: for(unsigned ii = 0; ii < n_in; ii++) {
        typename data_T::value_type cache = data_window[ii].read();
        ProductOutLoop: for(unsigned jj = 0; jj < n_out; jj++) {
            int index = ii * n_out + jj;
            //out_pixel[jj] += (res_T) cache * weights[index];
            mult[index] = cache * weights[index];
        }
    }

    AccumLoop: for(unsigned index = 0, jj = 0; index < n_in * n_out; index++, jj++) {
        #pragma HLS UNROLL
        if(jj == n_out) {
            jj = 0;
        }
        acc[jj] += mult[index];
    }

    CastLoop: for(unsigned jj = 0; jj < n_out; jj++) {
        #pragma HLS UNROLL
        if (CONFIG_T::out_pack_factor == 1) {
            res_pack[jj] = (typename res_T::value_type) acc[jj];
        } else {
            res_pack[outputs_ready * n_out + jj] = (typename res_T::value_type) acc[jj];
        }

    }

    if (CONFIG_T::out_pack_factor == 1) {
        res.write(res_pack);
    } else {
        if (outputs_ready == CONFIG_T::out_pack_factor - 1) {
            res.write(res_pack);
            outputs_ready = 0;
        } else {
            outputs_ready++;
        }
    }

}

template<class data_T, class res_T, typename CONFIG_T>
void compute_output(
    hls::stream<typename data_T::value_type> data_window[CONFIG_T::filt_height * CONFIG_T::filt_width * CONFIG_T::n_chan],
    hls::stream<res_T> &res,
    res_T &res_pack,
    unsigned &outputs_ready,
    typename CONFIG_T::weight_t weights[CONFIG_T::filt_height * CONFIG_T::filt_width * CONFIG_T::n_chan * CONFIG_T::n_filt],
    typename CONFIG_T::bias_t biases[CONFIG_T::n_filt],
    ap_uint<CONFIG_T::filt_height * CONFIG_T::filt_width> pixel_idx[CONFIG_T::in_pack_factor]
) {
    PixelLoop: for (unsigned p = 0; p < CONFIG_T::in_pack_factor; p++) {
        #pragma HLS PIPELINE
        if (pixel_idx[p][CONFIG_T::filt_height * CONFIG_T::filt_width - 1]) {
            mult_buffer<data_T, res_T, CONFIG_T>(data_window, res, res_pack, outputs_ready, weights, biases);
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
    assert(CONFIG_T::stride_height == 1 && CONFIG_T::stride_width == 1);
    //TODO add support for 'same' padding

    hls::stream<typename data_T::value_type> data_window[CONFIG_T::filt_height * CONFIG_T::filt_width * CONFIG_T::n_chan];
    const int win_depth = CONFIG_T::filt_height * CONFIG_T::out_width;
    for (unsigned i_out = 0; i_out < CONFIG_T::filt_height * CONFIG_T::filt_width * CONFIG_T::n_chan; i_out++) {
        #pragma HLS STREAM variable=data_window[i_out] depth=win_depth
    }

    #pragma HLS ARRAY_PARTITION variable=weights complete
    #pragma HLS ARRAY_PARTITION variable=biases complete
    //#pragma HLS ARRAY_PARTITION variable=CONFIG_T::pixels complete
    //#pragma HLS ARRAY_PARTITION variable=CONFIG_T::pixels->out_idx complete dim=0

    res_T res_pack;
    #pragma HLS DATA_PACK variable=res_pack
    unsigned outputs_ready = 0;

    ap_uint<CONFIG_T::filt_height * CONFIG_T::filt_width> pixel_idx[CONFIG_T::in_pack_factor];
    #pragma HLS ARRAY_PARTITION variable=pixel_idx complete

    ReadInputHeight: for (unsigned i_ih = 0; i_ih < CONFIG_T::in_height; i_ih++) {
        ReadInputWidth: for (unsigned i_iw = 0; i_iw < CONFIG_T::in_width / CONFIG_T::in_pack_factor; i_iw++) {
            #pragma HLS LOOP_FLATTEN
            //#pragma HLS PIPELINE // This works, but uses far more LUTs
            //#pragma HLS INLINE region
            compute_scaled_indices<CONFIG_T>(i_ih, i_iw, pixel_idx);
            fill_buffer<data_T, CONFIG_T>(data.read(), pixel_idx, data_window);
            compute_output<data_T, res_T, CONFIG_T>(data_window, res, res_pack, outputs_ready, weights, biases, pixel_idx);
        }
    }
}

}
#endif
