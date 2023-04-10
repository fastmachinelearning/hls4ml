#ifndef NNET_DENSE_RESOURCE_H_
#define NNET_DENSE_RESOURCE_H_

#include "hls_stream.h"
#include "nnet_common.h"
#include "nnet_mult.h"
#include <assert.h>
#include <math.h>

namespace nnet {

template <class data_T, class res_T, typename CONFIG_T>
void dense_resource_rf_leq_nin(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_out],
                               typename CONFIG_T::weight_t weights[CONFIG_T::n_in * CONFIG_T::n_out],
                               typename CONFIG_T::bias_t biases[CONFIG_T::n_out]) {

    const int multiplier_limit = DIV_ROUNDUP(CONFIG_T::n_in * CONFIG_T::n_out, CONFIG_T::reuse_factor);
    const int block_factor = DIV_ROUNDUP(CONFIG_T::n_in * CONFIG_T::n_out, CONFIG_T::reuse_factor);
    const int multscale = multiplier_limit / CONFIG_T::n_out;

    assert((multiplier_limit % CONFIG_T::n_out == 0 || CONFIG_T::reuse_factor >= CONFIG_T::n_in) &&
           "The current Reuse Factor is not allowed");
    assert((multiplier_limit == block_factor) && "This function is correct only for RF <= N_IN");

    // Treating weights as 2d is required to make sure Vitis doesn't use urem cores to calculate indices.
    // Also, we don't apply ARRAY_RESHAPE pragma as Vitis figures this out on its own.
    typename CONFIG_T::weight_t(*weights_2d)[CONFIG_T::reuse_factor] =
        (typename CONFIG_T::weight_t(*)[CONFIG_T::reuse_factor])weights; // I got you now motherfucker!

    #pragma HLS ARRAY_PARTITION variable=biases complete

    typename CONFIG_T::accum_t acc[CONFIG_T::n_out];
    #pragma HLS ARRAY_PARTITION variable=acc complete

InitAccum:
    for (int iacc = 0; iacc < CONFIG_T::n_out; iacc++) {
        #pragma HLS UNROLL
        acc[iacc] = (typename CONFIG_T::accum_t)biases[iacc];
    }

ReuseLoop:
    for (int ir = 0; ir < CONFIG_T::reuse_factor; ir++) {
        #pragma HLS PIPELINE II=1 rewind

        int in_index = ir;
        int out_index = 0;
        int acc_step = 0;

    MultLoop:
        for (int im = 0; im < block_factor; im++) {
            #pragma HLS UNROLL

            acc[out_index] += static_cast<typename CONFIG_T::accum_t>(
                CONFIG_T::template product<data_T, typename CONFIG_T::weight_t>::product(data[in_index],
                                                                                         weights_2d[im][ir]));

            // Increment in_index
            in_index += CONFIG_T::reuse_factor;
            if (in_index >= CONFIG_T::n_in) {
                in_index = ir;
            }
            // Increment out_index
            if (acc_step + 1 >= multscale) {
                acc_step = 0;
                out_index++;
            } else {
                acc_step++;
            }
        }
    }

// Cast to "res_t" type
Result:
    for (int ires = 0; ires < CONFIG_T::n_out; ires++) {
        #pragma HLS UNROLL
        res[ires] = cast<data_T, res_T, CONFIG_T>(acc[ires]);
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void dense_resource_rf_gt_nin_rem0(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_out],
                                   typename CONFIG_T::weight_t weights[CONFIG_T::n_in * CONFIG_T::n_out],
                                   typename CONFIG_T::bias_t biases[CONFIG_T::n_out]) {

    const int multiplier_limit = DIV_ROUNDUP(CONFIG_T::n_in * CONFIG_T::n_out, CONFIG_T::n_in);
    const int block_factor = DIV_ROUNDUP(CONFIG_T::n_in * CONFIG_T::n_out, CONFIG_T::reuse_factor);

    assert((multiplier_limit % CONFIG_T::n_out == 0 || CONFIG_T::reuse_factor >= CONFIG_T::n_in) &&
           "The current Reuse Factor is not allowed");
    assert((CONFIG_T::reuse_factor > CONFIG_T::n_in && CONFIG_T::reuse_factor % CONFIG_T::n_in == 0) &&
           "This function is correct only for RF > N_IN && RF % N_IN == 0");

    // Treating weights as 2d is required to make sure Vitis doesn't use urem cores to calculate indices.
    // Also, we don't apply ARRAY_RESHAPE pragma as Vitis figures this out on its own.
    typename CONFIG_T::weight_t(*weights_2d)[CONFIG_T::reuse_factor] =
        (typename CONFIG_T::weight_t(*)[CONFIG_T::reuse_factor])weights;

    #pragma HLS ARRAY_PARTITION variable=biases complete

    typename CONFIG_T::accum_t acc[CONFIG_T::n_out];
    #pragma HLS ARRAY_PARTITION variable=acc complete

InitAccum:
    for (int iacc = 0; iacc < CONFIG_T::n_out; iacc++) {
        #pragma HLS UNROLL
        acc[iacc] = (typename CONFIG_T::accum_t)biases[iacc];
    }

    int in_index = 0;
    int out_index;
    int outstep = 0;
    const int outscale = CONFIG_T::reuse_factor / CONFIG_T::n_in;

    int outidx[CONFIG_T::reuse_factor];
IndexLoop:
    for (int ir = 0; ir < CONFIG_T::reuse_factor; ir++) {
        outidx[ir] = outstep;
        if ((ir + 1) % CONFIG_T::n_in == 0) {
            outstep++;
        }
    }

ReuseLoop:
    for (int ir = 0; ir < CONFIG_T::reuse_factor; ir++) {
        #pragma HLS PIPELINE II=1 rewind

        out_index = outidx[ir] /*outstep*/;

    MultLoop:
        for (int im = 0; im < block_factor; im++) {
            #pragma HLS UNROLL

            acc[out_index] += static_cast<typename CONFIG_T::accum_t>(
                CONFIG_T::template product<data_T, typename CONFIG_T::weight_t>::product(data[in_index],
                                                                                         weights_2d[im][ir]));

            out_index += outscale;
        }

        in_index++;
        if (in_index >= CONFIG_T::n_in) {
            in_index = 0;
            // outstep++; // This causes a huge increase in scheduling and RTL generation times, hence the above workaround.
        }
    }

// Cast to "res_t" type
Result:
    for (int ires = 0; ires < CONFIG_T::n_out; ires++) {
        #pragma HLS UNROLL
        res[ires] = cast<data_T, res_T, CONFIG_T>(acc[ires]);
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void dense_resource_rf_gt_nin(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_out],
                              typename CONFIG_T::weight_t weights[CONFIG_T::n_in * CONFIG_T::n_out],
                              typename CONFIG_T::bias_t biases[CONFIG_T::n_out]) {

    const int multiplier_limit = CONFIG_T::n_out;
    const int block_factor = DIV_ROUNDUP(CONFIG_T::n_in * CONFIG_T::n_out, CONFIG_T::reuse_factor);

    assert((multiplier_limit % CONFIG_T::n_out == 0 || CONFIG_T::reuse_factor >= CONFIG_T::n_in) &&
           "The current Reuse Factor is not allowed");
    assert((CONFIG_T::reuse_factor > CONFIG_T::n_in) && "This function is correct only for RF > N_IN");

    // Treating weights as 2d is required to make sure Vitis doesn't use urem cores to calculate indices.
    // Also, we don't apply ARRAY_RESHAPE pragma as Vitis figures this out on its own.
    typename CONFIG_T::weight_t(*weights_2d)[CONFIG_T::reuse_factor] =
        (typename CONFIG_T::weight_t(*)[CONFIG_T::reuse_factor])weights;

    #pragma HLS ARRAY_PARTITION variable=biases complete

    typename CONFIG_T::accum_t acc[CONFIG_T::n_out];
    #pragma HLS ARRAY_PARTITION variable=acc complete

InitAccum:
    for (int iacc = 0; iacc < CONFIG_T::n_out; iacc++) {
        #pragma HLS UNROLL
        acc[iacc] = (typename CONFIG_T::accum_t)biases[iacc];
    }

ReuseLoop:
    for (int ir = 0; ir < CONFIG_T::reuse_factor; ir++) {
        #pragma HLS PIPELINE II=1 rewind
        typename CONFIG_T::accum_t tmpmult[block_factor];
        #pragma HLS ARRAY_PARTITION variable=tmpmult complete

    MultLoop:
        for (int im = 0; im < block_factor; im++) {
            #pragma HLS UNROLL
            int w_index = ir + CONFIG_T::reuse_factor * im;
            int in_index = w_index % CONFIG_T::n_in; // As of Vitis HLS 2022.1, this still results in urem core being used.
            tmpmult[im] =
                CONFIG_T::template product<data_T, typename CONFIG_T::weight_t>::product(data[in_index], weights_2d[im][ir]);
        }

        typename CONFIG_T::accum_t mult[multiplier_limit];
        #pragma HLS ARRAY_PARTITION variable=mult complete

    ResetMult:
        for (int imult = 0; imult < multiplier_limit; imult++) {
            #pragma HLS UNROLL
            mult[imult] = 0;
        }

    AccumLoop1:
        for (int im = 0; im < block_factor; im++) {
            #pragma HLS UNROLL
            int w_index = ir + CONFIG_T::reuse_factor * im;
            int out_index = w_index / CONFIG_T::n_in;
            if (out_index >= multiplier_limit)
                continue; // check out of bounds
            mult[out_index] += tmpmult[im];
        }

    AccumLoop2:
        for (int im = 0; im < multiplier_limit; im++) {
            #pragma HLS UNROLL
            acc[im] += mult[im]; // If RF > N_IN then multiplier_limit == n_out
        }
    }

// Cast to "res_t" type
Result:
    for (int ires = 0; ires < CONFIG_T::n_out; ires++) {
        #pragma HLS UNROLL
        res[ires] = cast<data_T, res_T, CONFIG_T>(acc[ires]);
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void dense_resource(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_out],
                    typename CONFIG_T::weight_t weights[CONFIG_T::n_in * CONFIG_T::n_out],
                    typename CONFIG_T::bias_t biases[CONFIG_T::n_out]) {

    #pragma HLS INLINE recursive

    if (CONFIG_T::reuse_factor <= CONFIG_T::n_in) {
        dense_resource_rf_leq_nin<data_T, res_T, CONFIG_T>(data, res, weights, biases);
    } else if (CONFIG_T::reuse_factor % CONFIG_T::n_in == 0) {
        dense_resource_rf_gt_nin_rem0<data_T, res_T, CONFIG_T>(data, res, weights, biases);
    } else {
        dense_resource_rf_gt_nin<data_T, res_T, CONFIG_T>(data, res, weights, biases);
    }
}

} // namespace nnet

#endif
