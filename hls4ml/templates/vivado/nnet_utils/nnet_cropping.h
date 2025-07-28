#ifndef NNET_CROPPING_H_
#define NNET_CROPPING_H_

#include <math.h>

namespace nnet {

struct cropping1d_config {
    static const unsigned n_chan = 10;
    static const unsigned in_width = 10;
    static const unsigned out_width = 10;
    static const unsigned crop_left = 0;
    static const unsigned crop_right = 0;
};

// no need for channel first for 1D cropping (no keras equivalent)
template <class data_T, class res_T, typename CONFIG_T>
void cropping1d_cl(data_T data[CONFIG_T::n_chan * CONFIG_T::in_width], res_T res[CONFIG_T::n_chan * CONFIG_T::out_width]) {
    #pragma HLS PIPELINE

    // Skip cropped input from left
    data += CONFIG_T::crop_left * CONFIG_T::n_chan;

    // Fill upto out_width (implicit cropping from right)
    for (int i = 0; i < CONFIG_T::out_width; i++) {
        for (int j = 0; j < CONFIG_T::n_chan; j++) {
            *(res++) = (res_T) * (data++);
        }
    }
}

struct cropping2d_config {
    static const unsigned n_chan = 10;
    static const unsigned in_height = 10;
    static const unsigned in_width = 10;
    static const unsigned out_height = 10;
    static const unsigned out_width = 10;
    static const unsigned crop_top = 0;
    static const unsigned crop_bottom = 0;
    static const unsigned crop_left = 0;
    static const unsigned crop_right = 0;
};

template <class data_T, class res_T, typename CONFIG_T>
void cropping2d_cf(data_T data[CONFIG_T::n_chan * CONFIG_T::in_height * CONFIG_T::in_width],
                   res_T res[CONFIG_T::n_chan * CONFIG_T::out_height * CONFIG_T::out_width]) {
    #pragma HLS PIPELINE

    for (int k = 0; k < CONFIG_T::n_chan; k++) { // channels first
        // Skip current channel data from top and left
        data_T *data_ptr = data + k * CONFIG_T::in_height * CONFIG_T::in_width + CONFIG_T::crop_top * CONFIG_T::in_width +
                           CONFIG_T::crop_left;

        // Fill upto out_height and out_width
        for (int i = 0; i < CONFIG_T::out_height; i++) {
            data_T *row_ptr = data_ptr + i * CONFIG_T::in_width;
            for (int j = 0; j < CONFIG_T::out_width; j++) {
                *(res++) = (res_T) * (row_ptr++);
            }
        }
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void cropping2d_cl(data_T data[CONFIG_T::n_chan * CONFIG_T::in_height * CONFIG_T::in_width],
                   res_T res[CONFIG_T::n_chan * CONFIG_T::out_height * CONFIG_T::out_width]) {
    #pragma HLS PIPELINE

    for (int i = 0; i < CONFIG_T::out_height; i++) {
        int in_row = i + CONFIG_T::crop_top;
        for (int j = 0; j < CONFIG_T::out_width; j++) {
            int in_col = j + CONFIG_T::crop_left;

            data_T *data_ptr = data + (in_row * CONFIG_T::in_width + in_col) * CONFIG_T::n_chan;
            for (int k = 0; k < CONFIG_T::n_chan; k++) { // channels last
                *(res++) = (res_T) * (data_ptr++);
            }
        }
    }
}

} // namespace nnet

#endif
