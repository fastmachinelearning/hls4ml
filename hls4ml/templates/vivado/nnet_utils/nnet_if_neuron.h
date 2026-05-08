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
    static const unsigned window_size = 0;
    static const bool threshold_is_vector = false;
    static constexpr float threshold = 1.0;
    static const snn_reset_mode reset_mode = snn_reset_mode::subtract;
    typedef float threshold_t;
    typedef float membrane_t;
};

template <class data_T, class res_T, typename CONFIG_T>
void if_neuron(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_out],
               const typename CONFIG_T::threshold_t threshold_vec[CONFIG_T::n_out]) {
    #pragma HLS PIPELINE II=1

    // Static state persists across calls until the configured time window ends.
    static typename CONFIG_T::membrane_t mem[CONFIG_T::n_out];
    #pragma HLS ARRAY_PARTITION variable=mem complete
    static unsigned ts = 0;

    for (unsigned i = 0; i < CONFIG_T::n_out; i++) {
        #pragma HLS UNROLL
        typename CONFIG_T::threshold_t threshold =
            CONFIG_T::threshold_is_vector ? threshold_vec[i] : (typename CONFIG_T::threshold_t)CONFIG_T::threshold;
        typename CONFIG_T::membrane_t v = mem[i] + (typename CONFIG_T::membrane_t)data[i];
        bool spike = (v >= threshold);
        if (spike) {
            if (CONFIG_T::reset_mode == snn_reset_mode::subtract) {
                v = v - threshold;
            } else {
                v = 0;
            }
            res[i] = 1;
        } else {
            res[i] = 0;
        }
        mem[i] = v;
    }

    if (CONFIG_T::window_size > 0) {
        ts++;
        if (ts >= CONFIG_T::window_size) {
            ts = 0;
            for (unsigned i = 0; i < CONFIG_T::n_out; i++) {
                #pragma HLS UNROLL
                mem[i] = 0;
            }
        }
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void if_neuron(hls::stream<data_T> &data_stream, hls::stream<res_T> &res_stream,
               const typename CONFIG_T::threshold_t threshold_vec[CONFIG_T::n_out]) {
    #pragma HLS PIPELINE II=1

    // Static state persists across calls until the configured time window ends.
    static typename CONFIG_T::membrane_t mem[CONFIG_T::n_out];
    #pragma HLS ARRAY_PARTITION variable=mem complete
    static unsigned ts = 0;

    data_T in_pack = data_stream.read();
    res_T out_pack;
    PRAGMA_DATA_PACK(out_pack)

    for (unsigned i = 0; i < CONFIG_T::n_out; i++) {
        #pragma HLS UNROLL
        typename CONFIG_T::threshold_t threshold =
            CONFIG_T::threshold_is_vector ? threshold_vec[i] : (typename CONFIG_T::threshold_t)CONFIG_T::threshold;
        typename CONFIG_T::membrane_t v = mem[i] + (typename CONFIG_T::membrane_t)in_pack[i];
        bool spike = (v >= threshold);
        if (spike) {
            if (CONFIG_T::reset_mode == snn_reset_mode::subtract) {
                v = v - threshold;
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

    if (CONFIG_T::window_size > 0) {
        ts++;
        if (ts >= CONFIG_T::window_size) {
            ts = 0;
            for (unsigned i = 0; i < CONFIG_T::n_out; i++) {
                #pragma HLS UNROLL
                mem[i] = 0;
            }
        }
    }
}

} // namespace nnet

#endif
