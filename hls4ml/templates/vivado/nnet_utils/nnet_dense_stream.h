#ifndef NNET_STREAM_LAYER_H_
#define NNET_STREAM_LAYER_H_

#include "nnet_common.h"
#include "nnet_types.h"
#include "hls_stream.h"
#include <math.h>
#include <assert.h>

namespace nnet {

template<class data_T, class res_T, typename CONFIG_T>
void dense_latency(
    hls::stream<data_T> &data,
    hls::stream<res_T> &res,
    typename CONFIG_T::weight_t  weights[CONFIG_T::n_in*CONFIG_T::n_out],
    typename CONFIG_T::bias_t    biases[CONFIG_T::n_out])
{
    data_T in_cache;
    res_T out_cache;

    #pragma HLS ARRAY_PARTITION variable=weights cyclic factor=CONFIG_T::n_out
    #pragma HLS ARRAY_PARTITION variable=biases complete

    hls::stream<res_T> out_s;
    #pragma HLS STREAM variable=out_s depth=CONFIG_T::n_out

    constexpr unsigned in_pack_factor = data_T::size;
    constexpr unsigned out_pack_factor = res_T::size;

    BiasLoop: for (int i = 0; i < CONFIG_T::n_out / out_pack_factor; i++) {
        #pragma HLS PIPELINE

        res_T out_part;
        BiasInner: for (int j = 0; j < out_pack_factor; j++) {
            #pragma HLS UNROLL
            out_part[j] = biases[i * out_pack_factor + j];
        }
        out_s.write(out_part);
    }

    int w_idx = 0;

    // Do the matrix-multiply
    Product1: for(int i_in = 0; i_in < CONFIG_T::n_in / in_pack_factor; i_in++) {
        in_cache = data.read();
        InPackLoop: for(int i_in_pack = 0; i_in_pack < in_pack_factor; i_in_pack++) {
            Product2: for(int i_out = 0; i_out < CONFIG_T::n_out / out_pack_factor; i_out++) {
                #pragma HLS PIPELINE II=3

                out_cache = out_s.read();
                OutPackLoop: for(int i_out_pack = 0; i_out_pack < out_pack_factor; i_out_pack++) {
                    #pragma HLS UNROLL
                    out_cache[i_out_pack] += in_cache[i_in_pack] * weights[w_idx++];
                }
                out_s.write(out_cache);
            }
        }
    }

    CastResult: for(int i_res = 0; i_res < CONFIG_T::n_out / out_pack_factor; i_res++) {
        #pragma HLS PIPELINE
        res.write(out_s.read());
    }
}

template<class data_T, class res_T, typename CONFIG_T>
void dense_latency_nout_pack(
    hls::stream<data_T> &data,
    hls::stream<res_T> &res,
    typename CONFIG_T::weight_t  weights[CONFIG_T::n_in*CONFIG_T::n_out],
    typename CONFIG_T::bias_t    biases[CONFIG_T::n_out])
{
    data_T in_cache;
    res_T out_cache;

    #pragma HLS ARRAY_PARTITION variable=weights cyclic factor=CONFIG_T::n_out
    #pragma HLS ARRAY_PARTITION variable=biases complete

    hls::stream<res_T> out_s;
    #pragma HLS STREAM variable=out_s depth=CONFIG_T::n_out

    constexpr unsigned in_pack_factor = data_T::size;
    constexpr unsigned out_pack_factor = res_T::size;

    res_T out_part;
    BiasLoop: for (int i = 0; i < out_pack_factor; i++) {
        #pragma HLS UNROLL
        out_part[i] = biases[i];
    }
    out_s.write(out_part);

    int w_idx = 0;

    // Do the matrix-multiply
    ProductLoop: for(int i_in = 0; i_in < CONFIG_T::n_in / in_pack_factor; i_in++) {
        in_cache = data.read();
        InPackLoop: for(int i_in_pack = 0; i_in_pack < in_pack_factor; i_in_pack++) {
            #pragma HLS PIPELINE II=3

            out_cache = out_s.read();
            OutPackLoop: for(int i_out_pack = 0; i_out_pack < out_pack_factor; i_out_pack++) {
                #pragma HLS UNROLL
                out_cache[i_out_pack] += in_cache[i_in_pack] * weights[w_idx++];
            }
            out_s.write(out_cache);
        }
    }

    res.write(out_s.read());
}

template<class data_T, class res_T, typename CONFIG_T>
void dense(
    hls::stream<data_T> &data,
    hls::stream<res_T> &res,
    typename CONFIG_T::weight_t  weights[CONFIG_T::n_in*CONFIG_T::n_out],
    typename CONFIG_T::bias_t    biases[CONFIG_T::n_out])
{
    if (CONFIG_T::strategy == latency) {
        if (CONFIG_T::n_out == res_T::size) {
    		dense_latency_nout_pack<data_T, res_T, CONFIG_T>(data, res, weights, biases);
    	} else {
    		dense_latency<data_T, res_T, CONFIG_T>(data, res, weights, biases);
    	}
    } else {
        assert("Streaming for resource strategy is not implemented yet" && false);
        //dense_resource<data_T, res_T, CONFIG_T>(data, res, weights, biases);
    }
}


}

#endif
