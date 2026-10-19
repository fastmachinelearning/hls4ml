#ifndef NNET_INSTANCENORM_H_
#define NNET_INSTANCENORM_H_

#include "nnet_common.h"
#include <math.h>

#include "hls_math.h"

namespace nnet {

struct instancenorm_config {
    // Internal data type definitions
    typedef float bias_t;
    typedef float scale_t;
    typedef float accum_t;

    // Layer Sizes
    static const unsigned n_in = 64;
    static const unsigned n_filt = 8;
    static const unsigned n_spatial = 8;
    static constexpr float epsilon = 1e-3;

    // Resource reuse info
    static const unsigned io_type = io_parallel;
    static const unsigned reuse_factor = 1;

    template <class x_T, class y_T> using product = nnet::product::mult<x_T, y_T>;
};

// Instance normalization normalizes each channel of the input using the statistics of the current
// sample, computed over the spatial positions. The input is channels last, i.e., the element at
// spatial position j of channel c is stored at index j * n_filt + c.
template <class data_T, class res_T, typename CONFIG_T>
void instancenormalize(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in],
                       typename CONFIG_T::scale_t scale[CONFIG_T::n_filt],
                       typename CONFIG_T::bias_t bias[CONFIG_T::n_filt]) {
    #pragma HLS ARRAY_PARTITION variable=scale complete
    #pragma HLS ARRAY_PARTITION variable=bias complete

    typename CONFIG_T::accum_t sums[CONFIG_T::n_filt];
    typename CONFIG_T::accum_t sums_sq[CONFIG_T::n_filt];
    typename CONFIG_T::accum_t mean[CONFIG_T::n_filt];
    typename CONFIG_T::accum_t inv_std[CONFIG_T::n_filt];
    #pragma HLS ARRAY_PARTITION variable=sums complete
    #pragma HLS ARRAY_PARTITION variable=sums_sq complete
    #pragma HLS ARRAY_PARTITION variable=mean complete
    #pragma HLS ARRAY_PARTITION variable=inv_std complete

    const typename CONFIG_T::accum_t k_inv = 1.0 / CONFIG_T::n_spatial;

InstanceNorm_Init:
    for (unsigned i = 0; i < CONFIG_T::n_filt; i++) {
        #pragma HLS UNROLL
        sums[i] = 0;
        sums_sq[i] = 0;
    }

// Accumulate the sum of values and the sum of squares of every channel
InstanceNorm_Sums:
    for (unsigned j = 0; j < CONFIG_T::n_spatial; j++) {
        #pragma HLS PIPELINE II=1
    InstanceNorm_Sums_Channels:
        for (unsigned i = 0; i < CONFIG_T::n_filt; i++) {
            #pragma HLS UNROLL
            typename CONFIG_T::accum_t val = static_cast<typename CONFIG_T::accum_t>(data[j * CONFIG_T::n_filt + i]);
            sums[i] += val;
            sums_sq[i] += val * val;
        }
    }

// Compute the per-channel statistics; the inverse standard deviation is computed in float
InstanceNorm_Stats:
    for (unsigned i = 0; i < CONFIG_T::n_filt; i++) {
        #pragma HLS UNROLL
        mean[i] =
            CONFIG_T::template product<typename CONFIG_T::accum_t, typename CONFIG_T::accum_t>::product(sums[i], k_inv);
        typename CONFIG_T::accum_t variance =
            CONFIG_T::template product<typename CONFIG_T::accum_t, typename CONFIG_T::accum_t>::product(sums_sq[i], k_inv) -
            mean[i] * mean[i];
        inv_std[i] =
            static_cast<typename CONFIG_T::accum_t>(1.0f / std::sqrt(static_cast<float>(variance) + CONFIG_T::epsilon));
    }

// Normalize the input and apply the affine transform
InstanceNorm_Result:
    for (unsigned j = 0; j < CONFIG_T::n_spatial; j++) {
        #pragma HLS PIPELINE II=1
    InstanceNorm_Result_Channels:
        for (unsigned i = 0; i < CONFIG_T::n_filt; i++) {
            #pragma HLS UNROLL
            unsigned index = j * CONFIG_T::n_filt + i;
            typename CONFIG_T::accum_t val = static_cast<typename CONFIG_T::accum_t>(data[index]);
            typename CONFIG_T::accum_t normalized = (val - mean[i]) * inv_std[i];
            res[index] = normalized * static_cast<typename CONFIG_T::accum_t>(scale[i]) +
                         static_cast<typename CONFIG_T::accum_t>(bias[i]);
        }
    }
}

} // namespace nnet

#endif
