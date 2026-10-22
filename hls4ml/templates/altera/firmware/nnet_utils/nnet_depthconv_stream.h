#ifndef NNET_DEPTHCONV_STREAM_H_
#define NNET_DEPTHCONV_STREAM_H_

#include "nnet_common.h"
#include "nnet_mult.h"

namespace nnet {

/*
 * void depthwise_product(kernel_window, res, weights, biases)
 *
 * Args:
 *   kernel_window - array of values from the input currently being convolved, ordered as [kernel_size][n_chan]
 *   res - array of n_filt outputs, ordered as [n_chan][depth_multiplier]
 *   weights - depthwise weights, ordered as [depth_multiplier][kernel_size][n_chan]
 *   biases - depthwise biases
 *
 * Each output filter depends only on its own input channel. The reuse factor sets how many filters share the same
 * multipliers; it is capped by the number of filters.
 */
template <class data_window_T, class res_T, typename CONFIG_T>
void depthwise_product(const data_window_T &kernel_window, res_T &res, const typename CONFIG_T::weight_t &weights,
                       const typename CONFIG_T::bias_t &biases) {
    constexpr unsigned depth_multiplier = CONFIG_T::n_filt / CONFIG_T::n_chan;
    constexpr unsigned rf = MIN(CONFIG_T::reuse_factor, CONFIG_T::n_filt);
    constexpr unsigned block_factor = DIV_ROUNDUP(CONFIG_T::n_filt, rf);

ReuseLoop:
    for (unsigned ir = 0; ir < rf; ir++) {
    FiltLoop:
        #pragma unroll
        for (unsigned im = 0; im < block_factor; im++) {
            const unsigned filt = ir + rf * im;
            if (filt >= CONFIG_T::n_filt)
                continue;

            const unsigned chan = filt / depth_multiplier;
            const unsigned dm = filt % depth_multiplier;

            typename CONFIG_T::accum_t acc = biases[filt];
        KernelLoop:
            #pragma unroll
            for (unsigned k = 0; k < CONFIG_T::kernel_size; k++) {
                acc += CONFIG_T::mult_config::
                    template product<typename data_window_T::value_type, typename CONFIG_T::weight_t::value_type>::product(
                        kernel_window[k * CONFIG_T::n_chan + chan],
                        weights[dm * CONFIG_T::kernel_size * CONFIG_T::n_chan + k * CONFIG_T::n_chan + chan]);
            }
            res[filt] = cast<typename CONFIG_T::accum_t, typename res_T::value_type, CONFIG_T>(acc);
        }
    }
}

} // namespace nnet

#endif
