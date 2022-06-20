#ifndef NNET_CONV1DTRANSPOSE_H_
#define NNET_CONV1DTRANSPOSE_H_

#include "nnet_common.h"
#include "nnet_conv1dtranspose_resource.h"
#include <cstdlib>

namespace nnet {

struct conv1dtranspose_config {
    // Internal data type definitions
    typedef float bias_t;
    typedef float weight_t;
    typedef float accum_t;

    // Convolutional parameters
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned in_width = 10;
    static const unsigned n_chan = 0;
    static const unsigned filt_width = 1;
    static const unsigned kernel_size = filt_width;
    static const unsigned stride_width = 1;
    static const unsigned dilation = 1;
    static const unsigned out_width = 10;

    static const unsigned reuse_factor = 1;
    static const bool store_weights_in_bram = false;
    static const unsigned n_zeros = 0;
};

template <class data_T, class res_T, typename CONFIG_T>
void conv_1d_transpose_cl(data_T data[CONFIG_T::in_width * CONFIG_T::n_chan],
                          res_T res[CONFIG_T::out_width * CONFIG_T::n_filt],
                          typename CONFIG_T::weight_t weights[CONFIG_T::stride_width]
                                                             [CONFIG_T::trfilt_width * CONFIG_T::n_filt * CONFIG_T::n_chan],
                          typename CONFIG_T::bias_t biases[CONFIG_T::n_filt]) {
    #pragma HLS INLINE region
    // for now, we are only adding resource strategy
    conv_1d_transpose_resource_cl<data_T, res_T, CONFIG_T>(data, res, weights, biases);
}

} // namespace nnet

#endif
