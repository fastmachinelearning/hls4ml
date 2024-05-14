#ifndef NNET_UNROLLED__H_
#define NNET_UNROLLED__H_

#include "nnet_common.h"
#include "nnet_helpers.h"

namespace nnet {
template <typename CONFIG_T, class data_T, class res_T>
typename std::enable_if<CONFIG_T::unrolled_fn != nullptr, void>::type dense(data_T data, res_T res) {
    #pragma HLS INLINE

    CONFIG_T::unrolled_fn(data, res);
}

template <typename CONFIG_T, class data_T, class res_T>
typename std::enable_if<CONFIG_T::mult_config::unrolled_fn != nullptr, void>::type
conv_1d(data_T data[CONFIG_T::in_width * CONFIG_T::n_chan], res_T res[CONFIG_T::out_width * CONFIG_T::n_filt]) {
    constexpr unsigned mult_n_in = CONFIG_T::n_chan * CONFIG_T::filt_width;
    constexpr unsigned mult_n_out = CONFIG_T::n_filt;

    data_T data_buf[CONFIG_T::n_pixels][mult_n_in];
    #pragma HLS ARRAY_PARTITION variable = data_buf complete dim = 0
    #pragma HLS INLINE

    res_T out_buf[mult_n_out];

PartitionLoop:
    for (int i_part = 0; i_part < CONFIG_T::n_partitions; i_part++) {
        #pragma HLS PIPELINE II = 1 rewind

        CONFIG_T::template fill_buffer<data_T, CONFIG_T>::fill_buffer(data, data_buf, i_part);

    PixelLoop:
        for (unsigned i_pxl = 0; i_pxl < CONFIG_T::n_pixels; i_pxl++) {
            #pragma HLS UNROLL

            // Do the matrix-multiply
            CONFIG_T::mult_config::unrolled_fn(data_buf[i_pxl], out_buf);

        Result:
            for (int i_res = 0; i_res < mult_n_out; i_res++) {
                #pragma HLS UNROLL
                *(res++) = out_buf[i_res];
            }
        }
    }
}

template <typename CONFIG_T, class data_T, class res_T>
typename std::enable_if<CONFIG_T::mult_config::unrolled_fn != nullptr, void>::type
conv_2d(data_T data[CONFIG_T::in_height * CONFIG_T::in_width * CONFIG_T::n_chan],
        res_T res[CONFIG_T::out_height * CONFIG_T::out_width * CONFIG_T::n_filt]) {
    constexpr unsigned mult_n_in = CONFIG_T::filt_height * CONFIG_T::filt_width * CONFIG_T::n_chan;
    constexpr unsigned mult_n_out = CONFIG_T::n_filt;

    data_T data_buf[CONFIG_T::n_pixels][mult_n_in];
    #pragma HLS ARRAY_PARTITION variable=data_buf complete dim = 0
    #pragma HLS INLINE

    res_T out_buf[mult_n_out];

PartitionLoop:
    for (int i_part = 0; i_part < CONFIG_T::n_partitions; i_part++) {
        #pragma HLS PIPELINE II=1 rewind

        CONFIG_T::template fill_buffer<data_T, CONFIG_T>::fill_buffer(data, data_buf, i_part);

    PixelLoop:
        for (unsigned i_pxl = 0; i_pxl < CONFIG_T::n_pixels; i_pxl++) {
            #pragma HLS UNROLL

            data_T cache;

            CONFIG_T::mult_config::unrolled_fn(data_buf[i_pxl], out_buf);

        Result:
            for (int i_res = 0; i_res < mult_n_out; i_res++) {
                #pragma HLS UNROLL
                *(res++) = out_buf[i_res];
            }
        }
    }
}

} // namespace nnet

#endif
