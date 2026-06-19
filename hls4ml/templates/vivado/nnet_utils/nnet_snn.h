#ifndef NNET_SNN_H_
#define NNET_SNN_H_

#include "nnet_common.h"

namespace nnet {

enum class snn_reset_mode { subtract, zero };
enum class snn_decision_rule {
    argmax_spike_count,
    first_to_threshold,
    threshold_then_argmax,
    binary_logit,
    argmax_membrane
};
enum class snn_readout_mode { spike, membrane };

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

struct lif_neuron_config {
    static const unsigned n_in = 1;
    static const unsigned n_out = 1;
    static const unsigned io_type = io_parallel;
    static const unsigned window_size = 0;
    static const bool beta_is_vector = false;
    static const bool threshold_is_vector = false;
    static constexpr float threshold = 1.0;
    static constexpr float beta = 0.9;
    static const snn_reset_mode reset_mode = snn_reset_mode::subtract;
    typedef float beta_t;
    typedef float threshold_t;
    typedef float membrane_t;
};

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
void lif_neuron(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_out],
                const typename CONFIG_T::beta_t beta_vec[CONFIG_T::n_out],
                const typename CONFIG_T::threshold_t threshold_vec[CONFIG_T::n_out]) {
    #pragma HLS PIPELINE II=1

    // Static state persists across calls until the configured time window ends.
    static typename CONFIG_T::membrane_t mem[CONFIG_T::n_out];
    #pragma HLS ARRAY_PARTITION variable=mem complete
    static unsigned ts = 0;

    for (unsigned i = 0; i < CONFIG_T::n_out; i++) {
        #pragma HLS UNROLL
        typename CONFIG_T::beta_t beta = CONFIG_T::beta_is_vector ? beta_vec[i] : (typename CONFIG_T::beta_t)CONFIG_T::beta;
        typename CONFIG_T::threshold_t threshold =
            CONFIG_T::threshold_is_vector ? threshold_vec[i] : (typename CONFIG_T::threshold_t)CONFIG_T::threshold;
        // LIF update: v[t] = beta * v[t-1] + input.
        typename CONFIG_T::membrane_t v =
            (typename CONFIG_T::membrane_t)(beta * mem[i]) + (typename CONFIG_T::membrane_t)data[i];
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

template <class data_T, typename CONFIG_T>
void update_snn_counts_array(data_T data[CONFIG_T::n_classes], unsigned counts[CONFIG_T::n_classes]) {
    for (unsigned i = 0; i < CONFIG_T::n_classes; i++) {
        #pragma HLS UNROLL
        counts[i] += (data[i] != 0) ? 1 : 0;
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

template <typename CONFIG_T> unsigned threshold_then_snn_argmax(unsigned counts[CONFIG_T::n_classes], unsigned fallback) {
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
unsigned update_snn_membrane_array(data_T data[CONFIG_T::n_classes],
                                   typename CONFIG_T::membrane_t mem[CONFIG_T::n_classes]) {
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

template <class data_T, class res_T, typename CONFIG_T>
typename res_T::value_type snn_membrane_readout_value(data_T data[CONFIG_T::n_classes],
                                                      typename CONFIG_T::membrane_t mem[CONFIG_T::n_classes]) {
    unsigned best = update_snn_membrane_array<data_T, CONFIG_T>(data, mem);
    if (CONFIG_T::decision_rule == snn_decision_rule::binary_logit) {
        return (typename res_T::value_type)(mem[1] - mem[0]);
    }
    return (typename res_T::value_type)best;
}

template <class data_T, class res_T, typename CONFIG_T>
typename res_T::value_type snn_spike_readout_value(data_T data[CONFIG_T::n_classes], unsigned counts[CONFIG_T::n_classes]) {
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

template <class data_T, class res_T, typename CONFIG_T> void snn_readout(data_T data[CONFIG_T::n_classes], res_T res[1]) {
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

} // namespace nnet

#endif
