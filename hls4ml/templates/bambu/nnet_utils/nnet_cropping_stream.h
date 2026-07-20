#ifndef NNET_CROPPING_STREAM_H_
#define NNET_CROPPING_STREAM_H_

#include "nnet_padding_stream.h" // fill_data function
#include <math.h>

namespace nnet {

template <class data_T, class res_T, typename CONFIG_T>
void cropping1d_cl(hls::stream<data_T> &data, hls::stream<res_T> &res) {
    //#pragma HLS PIPELINE

    // Discard left
    #pragma clang loop unroll(full)
    for (int i = 0; i < CONFIG_T::crop_left; i++) {
        data.read();
    }

    #pragma clang loop unroll(full)
    for (int i = 0; i < CONFIG_T::out_width; i++) {
        fill_data<data_T, res_T, CONFIG_T>(data, res);
    }

    // Discard right
    #pragma clang loop unroll(full)
    for (int i = 0; i < CONFIG_T::crop_right; i++) {
        data.read();
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void cropping2d_cl(hls::stream<data_T> &data, hls::stream<res_T> &res) {
    //#pragma HLS PIPELINE

    // Discard top rows
    #pragma clang loop unroll(full)
    for (int i = 0; i < CONFIG_T::crop_top; i++) {
        #pragma clang loop unroll(full)
        for (int j = 0; j < CONFIG_T::in_width; j++) {
            data.read();
        }
    }

    #pragma clang loop unroll(full)
    for (int i = 0; i < CONFIG_T::out_height; i++) {
        // Discard left columns
        #pragma clang loop unroll(full)
        for (int j = 0; j < CONFIG_T::crop_left; j++) {
            data.read();
        }
        #pragma clang loop unroll(full)
        for (int j = 0; j < CONFIG_T::out_width; j++) {
            fill_data<data_T, res_T, CONFIG_T>(data, res);
        }

        // Discard right columns
        #pragma clang loop unroll(full)
        for (int j = 0; j < CONFIG_T::crop_right; j++) {
            data.read();
        }
    }

    // Discard bottom rows
    #pragma clang loop unroll(full)
    for (int i = 0; i < CONFIG_T::crop_bottom; i++) {
        #pragma clang loop unroll(full)
        for (int j = 0; j < CONFIG_T::in_width; j++) {
            data.read();
        }
    }
}

} // namespace nnet

#endif
