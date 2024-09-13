#ifndef NNET_DENSE_SEQ_H_
#define NNET_DENSE_SEQ_H_

#include "hls_stream.h"
#include "nnet_common.h"
#include "nnet_helpers.h"
#include "nnet_mult.h"
#include <math.h>

namespace nnet {

template <class data_T, class res_T, typename CONFIG_T>
void dense_seq(data_T data[CONFIG_T::n_in * CONFIG_T::seq_len], res_T res[CONFIG_T::n_out * CONFIG_T::seq_len],
               typename CONFIG_T::weight_t weights[CONFIG_T::n_in * CONFIG_T::n_out],
               typename CONFIG_T::bias_t biases[CONFIG_T::n_out]) {
    #pragma HLS inline

    data_T in_val[CONFIG_T::n_in];
    #pragma HLS ARRAY_PARTITION variable=in_val complete

    if (CONFIG_T::strategy == nnet::latency) {
        for (int j = 0; j < CONFIG_T::seq_len; ++j) {
            #pragma HLS PIPELINE II=CONFIG_T::reuse_factor
            for (int i = 0; i < CONFIG_T::n_in; ++i) {
                #pragma HLS UNROLL
                in_val[i] = data[j * CONFIG_T::n_in + i];
            }
            dense_latency<data_T, res_T, CONFIG_T>(in_val, res + (CONFIG_T::n_out * j), weights, biases);
        }
    } else {
        for (int j = 0; j < CONFIG_T::seq_len; ++j) {
            #pragma HLS PIPELINE II=CONFIG_T::reuse_factor
            for (int i = 0; i < CONFIG_T::n_in; ++i) {
                #pragma HLS UNROLL
                in_val[i] = data[j * CONFIG_T::n_in + i];
            }
            dense_resource<data_T, res_T, CONFIG_T>(in_val, res + (CONFIG_T::n_out * j), weights, biases);
        }
    }
}

} // namespace nnet

#endif
