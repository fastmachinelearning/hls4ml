#ifndef NNET_SNN_READOUT_H_
#define NNET_SNN_READOUT_H_

#include "hls_stream.h"
#include "nnet_common.h"
#include "nnet_snn_common.h"

namespace nnet {

struct snn_readout_config {
    static const unsigned n_classes = 2;
    static const unsigned io_type = io_parallel;
    static const unsigned window_size = 1;
    static const unsigned class_threshold = 1;
    static constexpr float beta = 1.0;
    static const snn_readout_mode output_mode = snn_readout_mode::spike;
    static const snn_decision_rule decision_rule = snn_decision_rule::argmax_spike_count;
    typedef float membrane_t;
};

template <typename CONFIG_T> void reset_snn_counts(unsigned counts[CONFIG_T::n_classes]) {
    for (unsigned i = 0; i < CONFIG_T::n_classes; i++) {
        #pragma HLS UNROLL
        counts[i] = 0;
    }
}

template <typename CONFIG_T> void reset_snn_membrane(typename CONFIG_T::membrane_t mem[CONFIG_T::n_classes]) {
    for (unsigned i = 0; i < CONFIG_T::n_classes; i++) {
        #pragma HLS UNROLL
        mem[i] = 0;
    }
}

template <typename CONFIG_T> void advance_snn_count_window(unsigned &ts, unsigned counts[CONFIG_T::n_classes]) {
    ts++;
    if (ts >= CONFIG_T::window_size) {
        ts = 0;
        reset_snn_counts<CONFIG_T>(counts);
    }
}

template <typename CONFIG_T>
void advance_snn_membrane_window(unsigned &ts, typename CONFIG_T::membrane_t mem[CONFIG_T::n_classes]) {
    ts++;
    if (ts >= CONFIG_T::window_size) {
        ts = 0;
        reset_snn_membrane<CONFIG_T>(mem);
    }
}

template <typename res_T> void zero_snn_pack(res_T &out_pack) {
    for (unsigned i = 0; i < res_T::size; i++) {
        #pragma HLS UNROLL
        out_pack[i] = 0;
    }
}

template <class data_T, typename CONFIG_T>
void update_snn_counts_array(data_T data[CONFIG_T::n_classes], unsigned counts[CONFIG_T::n_classes]) {
    for (unsigned i = 0; i < CONFIG_T::n_classes; i++) {
        #pragma HLS UNROLL
        counts[i] += (data[i] != 0) ? 1 : 0;
    }
}

template <class data_T, typename CONFIG_T>
void update_snn_counts_pack(data_T in_pack, unsigned counts[CONFIG_T::n_classes]) {
    for (unsigned i = 0; i < CONFIG_T::n_classes; i++) {
        #pragma HLS UNROLL
        counts[i] += (in_pack[i] != 0) ? 1 : 0;
    }
}

template <typename CONFIG_T> unsigned argmax_snn_counts(unsigned counts[CONFIG_T::n_classes]) {
    unsigned best = 0;
    for (unsigned i = 1; i < CONFIG_T::n_classes; i++) {
        #pragma HLS UNROLL
        if (counts[i] > counts[best]) {
            best = i;
        }
    }
    return best;
}

template <typename CONFIG_T> unsigned first_snn_threshold(unsigned counts[CONFIG_T::n_classes], unsigned fallback) {
    unsigned best = fallback;
    for (unsigned i = 0; i < CONFIG_T::n_classes; i++) {
        #pragma HLS UNROLL
        if (counts[i] >= CONFIG_T::class_threshold) {
            best = i;
            break;
        }
    }
    return best;
}

template <typename CONFIG_T>
unsigned threshold_then_snn_argmax(unsigned counts[CONFIG_T::n_classes], unsigned fallback) {
    bool any_reached = false;
    for (unsigned i = 0; i < CONFIG_T::n_classes; i++) {
        #pragma HLS UNROLL
        any_reached |= (counts[i] >= CONFIG_T::class_threshold);
    }

    if (any_reached) {
        return first_snn_threshold<CONFIG_T>(counts, fallback);
    }

    return fallback;
}

template <class data_T, typename CONFIG_T>
unsigned update_snn_membrane_array(
    data_T data[CONFIG_T::n_classes], typename CONFIG_T::membrane_t mem[CONFIG_T::n_classes]
) {
    unsigned best = 0;
    for (unsigned i = 0; i < CONFIG_T::n_classes; i++) {
        #pragma HLS UNROLL
        typename CONFIG_T::membrane_t v =
            (typename CONFIG_T::membrane_t)((typename CONFIG_T::membrane_t)CONFIG_T::beta * mem[i]) +
            (typename CONFIG_T::membrane_t)data[i];
        mem[i] = v;
        if (i == 0 || v > mem[best]) {
            best = i;
        }
    }
    return best;
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
typename res_T::value_type snn_membrane_readout_value(
    data_T data[CONFIG_T::n_classes], typename CONFIG_T::membrane_t mem[CONFIG_T::n_classes]
) {
    unsigned best = update_snn_membrane_array<data_T, CONFIG_T>(data, mem);
    if (CONFIG_T::decision_rule == snn_decision_rule::binary_logit) {
        return (typename res_T::value_type)(mem[1] - mem[0]);
    }
    return (typename res_T::value_type)best;
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
typename res_T::value_type snn_spike_readout_value(
    data_T data[CONFIG_T::n_classes], unsigned counts[CONFIG_T::n_classes]
) {
    if (CONFIG_T::decision_rule == snn_decision_rule::argmax_spike_count) {
        unsigned best = 0;
        unsigned best_count = 0;
        for (unsigned i = 0; i < CONFIG_T::n_classes; i++) {
            #pragma HLS UNROLL
            unsigned c = counts[i] + ((data[i] != 0) ? 1 : 0);
            counts[i] = c;
            if (i == 0 || c > best_count) {
                best = i;
                best_count = c;
            }
        }
        return (typename res_T::value_type)best;
    }

    update_snn_counts_array<data_T, CONFIG_T>(data, counts);

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
void snn_readout(data_T data[CONFIG_T::n_classes], res_T res[1]) {
    #pragma HLS PIPELINE II=1

    // Counts and membrane values persist across calls within one readout window.
    static unsigned counts[CONFIG_T::n_classes];
    #pragma HLS ARRAY_PARTITION variable=counts complete
    static typename CONFIG_T::membrane_t mem[CONFIG_T::n_classes];
    #pragma HLS ARRAY_PARTITION variable=mem complete
    static unsigned ts = 0;

    if (CONFIG_T::output_mode == snn_readout_mode::membrane) {
        res[0] = snn_membrane_readout_value<data_T, res_T, CONFIG_T>(data, mem);
        advance_snn_membrane_window<CONFIG_T>(ts, mem);
        return;
    }

    res[0] = snn_spike_readout_value<data_T, res_T, CONFIG_T>(data, counts);
    advance_snn_count_window<CONFIG_T>(ts, counts);
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
