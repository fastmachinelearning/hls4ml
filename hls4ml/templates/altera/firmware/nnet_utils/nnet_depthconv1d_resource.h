#ifndef NNET_DEPTH_CONV1D_LATENCY_H_
#define NNET_DEPTH_CONV1D_LATENCY_H_

#include "nnet_common.h"
#include "nnet_conv1d_resource.h"
#include "nnet_mult.h"

namespace nnet {

template <class data_T, class res_T, typename CONFIG_T>
void depthwise_conv_1d_resource_cl(const data_T &data, res_T &res, const typename CONFIG_T::weight_t &weights,
                                   const typename CONFIG_T::bias_t &biases) {

    int depth_multiplier = CONFIG_T::n_filt / CONFIG_T::n_chan;
    [[intel::fpga_register]] int res_idx = 0;

    [[intel::fpga_register]] typename CONFIG_T::accum_t acc[CONFIG_T::out_width * CONFIG_T::n_filt];

DM_LOOP:
    #pragma unroll
    for (int dm = 0; dm < depth_multiplier; dm++) {

    WIDTH_LOOP:
        #pragma unroll
        for (int w = 0; w < CONFIG_T::out_width; w++) {

        CHAN_LOOP:
            #pragma unroll
            for (int c = 0; c < CONFIG_T::n_chan; c++) {

                res_idx = (w * CONFIG_T::n_filt) + (c * depth_multiplier) + dm;

                acc[res_idx] = biases[c * depth_multiplier + dm];

            KERNEL_W_LOOP:
                #pragma unroll
                for (int kw = 0; kw < CONFIG_T::filt_width; kw++) {

                    int w_in = w * CONFIG_T::stride_width + kw - CONFIG_T::pad_left;

                    if ((w_in >= 0) && (w_in < CONFIG_T::in_width)) {

                        acc[res_idx] += CONFIG_T::mult_config::
                            template product<typename data_T::value_type, typename CONFIG_T::weight_t::value_type>::product(
                                data[(w_in)*CONFIG_T::n_chan + c],
                                weights[(dm * CONFIG_T::filt_width * CONFIG_T::n_chan) + (kw * CONFIG_T::n_chan) + c]);
                    }
                }
            }
        }
    }

RESULT:
    #pragma unroll
    for (int ires = 0; ires < CONFIG_T::out_width * CONFIG_T::n_filt; ires++) {
        res[ires] = cast<typename CONFIG_T::accum_t, typename res_T::value_type, CONFIG_T>(acc[ires]);
    }
}
} // namespace nnet
#endif
