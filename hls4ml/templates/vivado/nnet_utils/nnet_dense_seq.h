#ifndef NNET_DENSE_SEQ_H_
#define NNET_DENSE_SEQ_H_

#include "nnet_common.h"
#include "nnet_mult.h"
#include "nnet_helpers.h"
#include "hls_stream.h"
#include <math.h>

namespace nnet {

template<class data_T, class res_T, typename CONFIG_T>
void dense_seq(
    data_T    data[CONFIG_T::n_in*CONFIG_T::seq_len],
    res_T     res[CONFIG_T::n_out*CONFIG_T::seq_len],
    typename CONFIG_T::weight_t  weights[CONFIG_T::n_in*CONFIG_T::n_out],
    typename CONFIG_T::bias_t    biases[CONFIG_T::n_out])
{
    #pragma HLS inline
    if (CONFIG_T::strategy == nnet::latency) {
        for (int j=0; j <CONFIG_T::seq_len; ++j){
            dense_latency<data_T, res_T, CONFIG_T>(data+(CONFIG_T::n_in*j), res+(CONFIG_T::n_out*j), weights, biases);
        }
    } else {
        for (int j=0; j <CONFIG_T::seq_len; ++j){
            dense_resource<data_T, res_T, CONFIG_T>(data+(CONFIG_T::n_in*j), res+(CONFIG_T::n_out*j), weights, biases);
        }
    }

}

}

#endif
