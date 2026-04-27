#ifndef NNET_IF_NEURON_H_
#define NNET_IF_NEURON_H_

#include "hls_stream.h"
#include "nnet_common.h"
#include "nnet_snn_common.h"

namespace nnet {

struct if_neuron_config {
    static const unsigned n_in = 1;
    static const unsigned n_out = 1;
    static const unsigned io_type = io_parallel;
    static constexpr float threshold = 1.0;
    static const snn_reset_mode reset_mode = snn_reset_mode::subtract;
};

template <class data_T, class res_T, typename CONFIG_T>
void if_neuron(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_out]) {
    #pragma HLS PIPELINE II=1

    static data_T mem[CONFIG_T::n_out];
    #pragma HLS ARRAY_PARTITION variable=mem complete

    for (unsigned i = 0; i < CONFIG_T::n_out; i++) {
        #pragma HLS UNROLL
        data_T v = mem[i] + data[i];
        bool spike = (v >= (data_T)CONFIG_T::threshold);
        if (spike) {
            if (CONFIG_T::reset_mode == snn_reset_mode::subtract) {
                v = v - (data_T)CONFIG_T::threshold;
            } else {
                v = 0;
            }
            res[i] = 1;
        } else {
            res[i] = 0;
        }
        mem[i] = v;
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void if_neuron(hls::stream<data_T> &data_stream, hls::stream<res_T> &res_stream) {
    #pragma HLS PIPELINE II=1

    static typename data_T::value_type mem[CONFIG_T::n_out];
    #pragma HLS ARRAY_PARTITION variable=mem complete

    data_T in_pack = data_stream.read();
    res_T out_pack;
    PRAGMA_DATA_PACK(out_pack)

    for (unsigned i = 0; i < CONFIG_T::n_out; i++) {
        #pragma HLS UNROLL
        typename data_T::value_type v = mem[i] + in_pack[i];
        bool spike = (v >= (typename data_T::value_type)CONFIG_T::threshold);
        if (spike) {
            if (CONFIG_T::reset_mode == snn_reset_mode::subtract) {
                v = v - (typename data_T::value_type)CONFIG_T::threshold;
            } else {
                v = 0;
            }
            out_pack[i] = 1;
        } else {
            out_pack[i] = 0;
        }
        mem[i] = v;
    }

    res_stream.write(out_pack);
}

} // namespace nnet

#endif
