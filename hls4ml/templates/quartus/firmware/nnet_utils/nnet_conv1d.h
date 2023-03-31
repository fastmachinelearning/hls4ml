#ifndef NNET_CONV1D_H_
#define NNET_CONV1D_H_

#include "nnet_common.h"
#include "nnet_conv1d_resource.h"

namespace nnet {

struct conv1d_config {
    // I/O sizes
    static const unsigned in_width = 10;
    static const unsigned out_width = 10;

    // Number of channels, filters
    static const unsigned n_chan = 1;
    static const unsigned n_filt = 1;

    // Original filter size
    static const unsigned filt_width = 1;
    static const unsigned kernel_size = filt_width;

    // Modified filter size (post-Wionograd transformation, if applied)
    static const unsigned impl_filt_height = 1;
    static const unsigned impl_filt_width = 1;

    // Padding, stride, dilation
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned stride_width = 1;
    static const unsigned dilation = 1;

    // Run-time Configuration
    static const unsigned n_zeros = 0;
    static const unsigned reuse_factor = 1;
    static const unsigned parallelisation_factor = 1;

    // TODO: BRAM Storage on Quartus
    static const bool store_weights_in_bram = false;

    // Internal data type definitions
    typedef float bias_t;
    typedef float weight_t;
    typedef float accum_t;
};

template <class data_T, class res_T, typename CONFIG_T>
void conv_1d_cl(data_T data[CONFIG_T::in_width * CONFIG_T::n_chan], res_T res[CONFIG_T::out_width * CONFIG_T::n_filt],
                const typename CONFIG_T::weight_t weights[CONFIG_T::filt_width * CONFIG_T::n_chan * CONFIG_T::n_filt],
                const typename CONFIG_T::bias_t biases[CONFIG_T::n_filt]) {
    conv_1d_resource_cl<data_T, res_T, CONFIG_T>(data, res, weights, biases);
}

template <class data_T, class res_T, typename CONFIG_T>
void pointwise_conv_1d_cl(data_T data[CONFIG_T::in_width * CONFIG_T::n_chan],
                          res_T res[CONFIG_T::out_width * CONFIG_T::n_filt],
                          const typename CONFIG_T::weight_t weights[CONFIG_T::n_chan * CONFIG_T::n_filt],
                          const typename CONFIG_T::bias_t biases[CONFIG_T::n_filt]) {
    assert(CONFIG_T::filt_width == 1);
    pointwise_conv_1d_resource_cl<data_T, res_T, CONFIG_T>(data, res, weights, biases);
}

} // namespace nnet

#endif
