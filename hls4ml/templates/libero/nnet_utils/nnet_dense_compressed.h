#ifndef NNET_COMPRESSED_LAYER_H_
#define NNET_COMPRESSED_LAYER_H_

#include "nnet_common.h"
#include "nnet_dense.h"
#include <hls/streaming.hpp>
#include <math.h>

namespace nnet {

template <typename CONFIG_T>
void fill_mult(typename CONFIG_T::index_t index, typename CONFIG_T::accum_t mult[CONFIG_T::n_out],
               typename CONFIG_T::accum_t weight) {
    #pragma HLS loop unroll
    for (unsigned k = 0; k < CONFIG_T::n_out; k++) {
        if (k == index)
            mult[k] += weight;
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void dense_compressed(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_out],
                      typename CONFIG_T::weight_t weights[CONFIG_T::n_nonzeros],
                      typename CONFIG_T::bias_t biases[CONFIG_T::n_out]) {

    #pragma HLS memory partition argument(biases) type(complete)
    const int multiplier_limit = DIV_ROUNDUP(CONFIG_T::n_nonzeros, CONFIG_T::reuse_factor);

    #pragma HLS memory partition variable(acc) type(complete)
    typename CONFIG_T::accum_t acc[CONFIG_T::n_out];
    //#pragma HLS ARRAY_RESHAPE   variable=weights block factor=multiplier_limit

    //#ifdef __VITIS_HLS__
    //    #pragma HLS AGGREGATE variable=weights
    //#else
    //    #pragma HLS data_pack variable=weights struct_level
    //#endif

InitAccum:
    #pragma HLS loop unroll
    for (unsigned i = 0; i < CONFIG_T::n_out; i++) {
        acc[i] = (typename CONFIG_T::accum_t)(biases[i]);
    }

    // Do the compressed matrix-multiply
    const int rufactor = CONFIG_T::reuse_factor;
ReuseLoop:
    for (unsigned ir = 0; ir < rufactor; ir++) {

        #pragma HLS memory partition variable(mult) type(complete)
        typename CONFIG_T::accum_t mult[CONFIG_T::n_out];

    ResetMult:
        #pragma HLS loop unroll
        for (int imult = 0; imult < CONFIG_T::n_out; imult++) {
            mult[imult] = 0;
        }

    CompressedMultLoop:
        #pragma HLS loop unroll
        for (unsigned im = 0; im < multiplier_limit; im++) {
            unsigned w = im * rufactor + ir;
            auto row = weights[w].row_index;
            auto col = weights[w].col_index;
            auto weight_cache = weights[w].weight;
            data_T data_cache = data[row];
            // mult[col] += weight_cache * data_cache;
            typename CONFIG_T::accum_t prod =
                CONFIG_T::template product<data_T, typename CONFIG_T::weight_t>::product(data_cache, weight_cache);
            fill_mult<CONFIG_T>(col, mult, prod);
        }

        for (int im = 0; im < CONFIG_T::n_out; im++) {
            acc[im] += mult[im];
        }
    }

// Cast to "res_t" type
ResultLoop:
    #pragma HLS loop unroll
    for (unsigned i = 0; i < CONFIG_T::n_out; i++) {
        // res[i] = (res_T) (acc[i]);
        res[i] = cast<data_T, res_T, CONFIG_T>(acc[i]);
    }
}

} // namespace nnet

#endif
