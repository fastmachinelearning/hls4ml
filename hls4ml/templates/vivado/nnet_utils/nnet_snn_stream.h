#ifndef NNET_SNN_STREAM_H_
#define NNET_SNN_STREAM_H_

#include "hls_stream.h"
#include "nnet_common.h"
#include "nnet_snn.h"

namespace nnet {

template <typename res_T> void zero_snn_pack(res_T &out_pack) {
    for (unsigned i = 0; i < res_T::size; i++) {
        #pragma HLS UNROLL
        out_pack[i] = 0;
    }
}

template <class data_T, typename CONFIG_T>
void update_snn_counts_pack(data_T in_pack, unsigned counts[CONFIG_T::n_classes]) {
    for (unsigned i = 0; i < CONFIG_T::n_classes; i++) {
        #pragma HLS UNROLL
        counts[i] += (in_pack[i] != 0) ? 1 : 0;
    }
}

template <class data_T, typename CONFIG_T>
unsigned update_snn_membrane_pack(data_T in_pack, typename CONFIG_T::membrane_t mem[CONFIG_T::n_classes]) {
    unsigned best = 0;
    for (unsigned i = 0; i < CONFIG_T::n_classes; i++) {
        #pragma HLS UNROLL
        typename CONFIG_T::membrane_t v =
            (typename CONFIG_T::membrane_t)((typename CONFIG_T::membrane_t)CONFIG_T::beta * mem[i]) +
            (typename CONFIG_T::membrane_t)in_pack[i];
        mem[i] = v;
        if (i == 0 || v > mem[best]) {
            best = i;
        }
    }
    return best;
}

template <class data_T, class res_T, typename CONFIG_T>
typename res_T::value_type snn_membrane_readout_value_pack(
    data_T in_pack, typename CONFIG_T::membrane_t mem[CONFIG_T::n_classes]
) {
    unsigned best = update_snn_membrane_pack<data_T, CONFIG_T>(in_pack, mem);
    if (CONFIG_T::decision_rule == snn_decision_rule::binary_logit) {
        return (typename res_T::value_type)(mem[1] - mem[0]);
    }
    return (typename res_T::value_type)best;
}

template <class data_T, class res_T, typename CONFIG_T>
typename res_T::value_type snn_spike_readout_value_pack(data_T in_pack, unsigned counts[CONFIG_T::n_classes]) {
    if (CONFIG_T::decision_rule == snn_decision_rule::argmax_spike_count) {
        unsigned best = 0;
        unsigned best_count = 0;
        for (unsigned i = 0; i < CONFIG_T::n_classes; i++) {
            #pragma HLS UNROLL
            unsigned c = counts[i] + ((in_pack[i] != 0) ? 1 : 0);
            counts[i] = c;
            if (i == 0 || c > best_count) {
                best = i;
                best_count = c;
            }
        }
        return (typename res_T::value_type)best;
    }

    update_snn_counts_pack<data_T, CONFIG_T>(in_pack, counts);

    if (CONFIG_T::decision_rule == snn_decision_rule::binary_logit) {
        return (typename res_T::value_type)((int)counts[1] - (int)counts[0]);
    }

    unsigned best = argmax_snn_counts<CONFIG_T>(counts);
    if (CONFIG_T::decision_rule == snn_decision_rule::first_to_threshold) {
        best = first_snn_threshold<CONFIG_T>(counts, best);
    } else if (CONFIG_T::decision_rule == snn_decision_rule::threshold_then_argmax) {
        best = threshold_then_snn_argmax<CONFIG_T>(counts, best);
    }

    return (typename res_T::value_type)best;
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

template <class data_T, class res_T, typename CONFIG_T>
void lif_neuron(hls::stream<data_T> &data_stream, hls::stream<res_T> &res_stream,
                const typename CONFIG_T::beta_t beta_vec[CONFIG_T::n_out],
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
        typename CONFIG_T::beta_t beta = CONFIG_T::beta_is_vector ? beta_vec[i] : (typename CONFIG_T::beta_t)CONFIG_T::beta;
        typename CONFIG_T::threshold_t threshold =
            CONFIG_T::threshold_is_vector ? threshold_vec[i] : (typename CONFIG_T::threshold_t)CONFIG_T::threshold;
        // LIF update: v[t] = beta * v[t-1] + input.
        typename CONFIG_T::membrane_t v =
            (typename CONFIG_T::membrane_t)(beta * mem[i]) + (typename CONFIG_T::membrane_t)in_pack[i];
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

template <class data_T, class res_T, typename CONFIG_T>
void snn_readout(hls::stream<data_T> &data_stream, hls::stream<res_T> &res_stream) {
    #pragma HLS PIPELINE II=1

    // Counts and membrane values persist across calls within one readout window.
    static unsigned counts[CONFIG_T::n_classes];
    #pragma HLS ARRAY_PARTITION variable=counts complete
    static typename CONFIG_T::membrane_t mem[CONFIG_T::n_classes];
    #pragma HLS ARRAY_PARTITION variable=mem complete
    static unsigned ts = 0;

    data_T in_pack = data_stream.read();

    if (CONFIG_T::output_mode == snn_readout_mode::membrane) {
        res_T out_pack;
        PRAGMA_DATA_PACK(out_pack)
        zero_snn_pack<res_T>(out_pack);
        out_pack[0] = snn_membrane_readout_value_pack<data_T, res_T, CONFIG_T>(in_pack, mem);
        res_stream.write(out_pack);
        advance_snn_membrane_window<CONFIG_T>(ts, mem);
        return;
    }

    res_T out_pack;
    PRAGMA_DATA_PACK(out_pack)
    zero_snn_pack<res_T>(out_pack);
    out_pack[0] = snn_spike_readout_value_pack<data_T, res_T, CONFIG_T>(in_pack, counts);
    res_stream.write(out_pack);
    advance_snn_count_window<CONFIG_T>(ts, counts);
}

} // namespace nnet

#endif
