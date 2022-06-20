#ifndef NNET_CONV2DTRANSPOSE_H
#define NNET_CONV2DTRANSPOSE_H

#include "nnet_common.h"
#include "nnet_conv2dtranspose_resource.h"
#include <cstdlib>

namespace nnet{

struct conv2dtranspose_config
{
    // Internal data type definitions
    typedef float bias_t;
    typedef float weight_t;
    typedef float accum_t;

    // Convolutional parameters
    static const unsigned pad_top = 0;
    static const unsigned pad_bottom = 0;
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned in_height = 10;
    static const unsigned in_width = 10;
    static const unsigned n_chan = 1;
    static const unsigned filt_height = 1;
    static const unsigned filt_width = 1;
    static const unsigned kernel_size = filt_height * filt_width;
    static const unsigned n_filt = 1;
    static const unsigned stride_height = 1;
    static const unsigned stride_width = 1;
    static const unsigned out_height = 10;
    static const unsigned out_width = 10;
    static const unsigned dilation_height = 1;
    static const unsigned dilation_width = 1;
    static const unsigned trfilt_height = 1;
    static const unsigned trfilt_width = 1;

    static const unsigned reuse_factor = 1;
    static const bool store_weights_in_bram = false;
    static const unsigned n_zeros = 0; // not used yet
};

template<class data_T, class res_T, typename CONFIG_T>
void conv_2d_transpose_cl(
    data_T data[CONFIG_T::in_height * CONFIG_T::in_width * CONFIG_T::n_chan],
    res_T  res[CONFIG_T::out_height * CONFIG_T::out_width * CONFIG_T::n_filt],
    typename CONFIG_T::weight_t weights[CONFIG_T::stride_height][CONFIG_T::stride_width][
        CONFIG_T::trfilt_height * CONFIG_T::trfilt_width * CONFIG_T::n_filt * CONFIG_T::n_chan
    ],
    typename CONFIG_T::bias_t   biases[CONFIG_T::n_filt]
)
{
    #pragma HLS INLINE region
    //only have resource strategy as of now
    conv_2d_transpose_resource_cl<data_T, res_T, CONFIG_T>(data, res, weights, biases);
}

}

#endif