#ifndef NNET_DENSE_STREAM_H_
#define NNET_DENSE_STREAM_H_

#include "nnet_common.h"
#include "nnet_types.h"
#include "ac_channel.h"
#include <math.h>
#include <assert.h>

namespace nnet {

template<class data_T, class res_T, typename CONFIG_T>
void dense_wrapper(
    data_T data[CONFIG_T::n_in],
    res_T  res[CONFIG_T::n_out],
    typename CONFIG_T::weight_t weights[CONFIG_T::n_in*CONFIG_T::n_out],
    typename CONFIG_T::bias_t   biases[CONFIG_T::n_out]
) {
    //#pragma HLS INLINE region
    if (CONFIG_T::strategy == nnet::latency) {
        //#pragma HLS PIPELINE II=CONFIG_T::reuse_factor
        dense_latency<data_T, res_T, CONFIG_T, CONFIG_T::reuse_factor>(data, res, weights, biases);
    } else {
        dense_resource<data_T, res_T, CONFIG_T>(data, res, weights, biases);
    }
}

#pragma hls_design
template<class data_T, class res_T, typename CONFIG_T>
void dense(
    ac_channel<data_T> &data_stream,
    ac_channel<res_T>  &res_stream,
    typename CONFIG_T::weight_t weights[CONFIG_T::n_in*CONFIG_T::n_out],
    typename CONFIG_T::bias_t   biases[CONFIG_T::n_out])
{
    typename data_T::value_type data[CONFIG_T::n_in];
    //#pragma HLS ARRAY_PARTITION variable=data complete

    typename res_T::value_type res[CONFIG_T::n_out];
    //#pragma HLS ARRAY_PARTITION variable=res complete

    if constexpr(CONFIG_T::n_in / data_T::size > 1) {
    #pragma hls_pipeline_init_interval 1
    }
    DataPrepare: for(int i_in = 0; i_in < CONFIG_T::n_in / data_T::size; i_in++) {
        if (CONFIG_T::n_in / data_T::size > 1) {
            //#pragma HLS PIPELINE
        }
        data_T data_pack = data_stream.read();
        #pragma hls_unroll
        DataPack: for (int i_pack = 0; i_pack < data_T::size; i_pack++) {
            //#pragma HLS UNROLL
            data[i_in * data_T::size + i_pack] = data_pack[i_pack];
        }
    }

    dense_wrapper<typename data_T::value_type, typename res_T::value_type, CONFIG_T>(data, res, weights, biases);

    if constexpr(CONFIG_T::n_out / res_T::size > 1) {
    #pragma hls_pipeline_init_interval 1
    }
    ResWrite: for(unsigned i_out = 0; i_out < CONFIG_T::n_out / res_T::size; i_out++) {
        if (CONFIG_T::n_out / res_T::size > 1) {
            //#pragma HLS PIPELINE
        }
        res_T res_pack;
        //#pragma HLS DATA_PACK variable=res_pack
        #pragma hls_unroll
        ResPack: for (int i_pack = 0; i_pack < res_T::size; i_pack++) {
            //#pragma HLS UNROLL
            res_pack[i_pack] = res[i_out * res_T::size + i_pack];
        }
        res_stream.write(res_pack);
    }
}

}

#endif
